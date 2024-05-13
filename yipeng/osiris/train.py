import os
import argparse
import json
import time
import numpy as np
import random
from pathlib import Path
import pickle
from copy import deepcopy
from torch import optim, nn
import torch

from models.simclr import SimCLR, Osiris
from models.continual_model import ContinualModel

from utils.loading_utils import get_stream_datasets, get_knn_data_loaders


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

def main_worker(gpu, args):
	args.rank = gpu
	torch.distributed.init_process_group(
		backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

	set_seed(args)

	dataset = get_stream_datasets(args)
	dataset.update_order(task=0) # call it here so that length of dataloader is correct
	sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=bool('iid' in args.order))
	assert args.batch_size % args.world_size == 0
	per_gpu_batch_size = args.batch_size // args.world_size
	train_loader = torch.utils.data.DataLoader(dataset, 
		batch_size=per_gpu_batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, 
		persistent_workers=True)

	###### DEBUG OVERFITTING TO MEMORY
	if args.debug_overfit:
		order_temp = args.order
		args.order = 'seen_iid'
		dataset_seen = get_stream_datasets(args)
		args.order = order_temp
	######

	#seed = int(random.random() * 100) # for continual models
	use_cl_wrapper = bool('continual' in args.model)

	steps_per_task = len(train_loader) * args.epochs
	if 'sliding_simclr' in args.model:
		model = SimCLR(args)
	elif 'continual_simclr' in args.model:
		model = Osiris(args)
	else:
		raise NotImplementedError

	if use_cl_wrapper:
		model = ContinualModel(args, model)

	# https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
	torch.cuda.set_device(args.rank)
	torch.cuda.empty_cache()

	model = model.cuda(args.rank)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)



	param_weights = []
	param_biases = []
	named_parameters = model.net.named_parameters() if use_cl_wrapper else model.named_parameters()
	for n, param in named_parameters:
		if param.ndim == 1: 
			param_biases.append(param)
		else: param_weights.append(param)
	parameters = [{'params': param_weights, 'lr': args.learning_rate_weights, 'weight_decay': args.weight_decay}, 
				{'params': param_biases, 'lr': args.learning_rate_biases, 'weight_decay': 0.0}]

	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])

	if args.optimizer == 'sgd':
		optimizer = optim.SGD(parameters, lr=args.learning_rate_weights, momentum=args.momentum, weight_decay=args.weight_decay)
	else:
		raise NotImplementedError('Optimizer not supported.')

	if args.rank == 0:
		memory_loaders, test_loaders = get_knn_data_loaders(args) # for knn monitoring

	start_time = time.time()

	loss_logs = [[]]
	eval_logs = []
	w_logs = [[]]
	w_loss = [args.w_simclr]
	for i in range(len(args.w_additional)):
		loss_logs.append([])
		w_loss.append(args.w_additional[i])
		w_logs.append([])

	w_loss = torch.tensor(w_loss, dtype=torch.float).cuda() 
	loss_totals = np.zeros(len(args.w_additional)+1, dtype=float)
	w_totals = np.zeros(len(args.w_additional)+1, dtype=float)

	seen_loss_logs = []
	mem_loss_logs = []

	backbone = model.module.net.backbone if use_cl_wrapper else model.module.backbone
	projector = model.module.net.projector if use_cl_wrapper else model.module.projector

	###################
	# Task outer loop #
	###################
	for task in range(args.num_tasks):
			
		if args.rank == 0:
			print('Task', task)

		if args.epochs > 1000 and task > 0:
			# first task only
			break

		if task != 0:
			dataset.update_order(task=task)
			# If persistent_workers is False then workers will create new copies of dataset then we don't need the following line
			train_loader = torch.utils.data.DataLoader(dataset, batch_size=per_gpu_batch_size, num_workers=args.num_workers, 
				pin_memory=True, sampler=sampler, persistent_workers=True)

		if args.debug_overfit:
			dataset_seen.update_order(task=task) # length is changing, so need to recreate sampler
			sampler_seen = torch.utils.data.distributed.DistributedSampler(dataset_seen, shuffle=True)
			train_loader_seen = torch.utils.data.DataLoader(dataset_seen, 
				batch_size=per_gpu_batch_size//3, num_workers=args.num_workers, pin_memory=True, sampler=sampler_seen, 
				persistent_workers=True)

		if 'continual' in args.model:
			model.module.update_model_states(task)

		num_epochs = args.epochs
		model.train()

		###################
		# Epoch inner loop #
		###################
		for epoch in range(num_epochs):
			sampler.set_epoch(epoch)
			if args.debug_overfit:
				sampler_seen.set_epoch(epoch)
				iter_seen = iter(train_loader_seen)

			for step, (img, y, labels) in enumerate(train_loader, start=epoch*len(train_loader)):

				global_step = step + task * steps_per_task + 1

				y1, y2 = y
				y1_inputs = y1.cuda(non_blocking=True)
				y2_inputs = y2.cuda(non_blocking=True)
				img = img.cuda(non_blocking=True)

				optimizer.zero_grad()

				if 'continual' in args.model:
					# seed = random.randint(1, 10000)
					# print(args.rank, seed)
					losses = model(y1, y2, img, labels=labels, task=task)
				else:
					losses, _ = model(y1_inputs, y2_inputs)

				loss_totals[0] += losses[0].item()
				w_totals[0] += w_loss[0].item()
				for li in range(len(losses[1])):
					loss_totals[li+1] += losses[1][li].item()
					w_totals[li+1] += w_loss[li+1].item()
				
				losses = torch.stack([losses[0]]+losses[1])

				loss = (w_loss[:losses.shape[0]] * losses).sum()


				loss.backward()
				optimizer.step()

				if args.rank == 0:
					if global_step % args.print_freq == 0:
						stats = dict(task=task,
									step=global_step,
									total=args.num_tasks*steps_per_task, 
									lr=optimizer.param_groups[0]['lr'],
									loss_ssl=loss_totals[0]/args.print_freq,
									loss_additional=(loss_totals[1:]/args.print_freq).tolist(),
									w_loss=(w_totals/args.print_freq).tolist(),
									# seen_loss=seen_loss[0],
									# mem_loss=mem_loss[0],
									time=int(time.time() - start_time))
						print(json.dumps(stats))
						for li in range(loss_totals.shape[0]):
							loss_logs[li].append(loss_totals[li]/args.print_freq)
							w_logs[li].append(w_totals[li]/args.print_freq)
						loss_totals[:] = 0
						w_totals[:] = 0

				if (args.num_tasks == 1 or args.epochs > 1000) and (global_step+1) % args.eval_freq == 0:
					if args.rank == 0:
						# IID or first task only
						save_count = (global_step + 1) // args.eval_freq - 1
						model.eval()
						eval_logs.append(eval_and_save(args, backbone, memory_loaders, test_loaders, savecount=task, use_cl_wrapper=use_cl_wrapper, 
							buf=None, projector=projector, model=model.module))
						model.train()
						torch.distributed.barrier()
					else:
						torch.distributed.barrier()
						

		if args.rank == 0:
			# end of task eval
			model.eval()
			buf = model.module.buffer.examples if args.replay_mode == 'mixup' or args.grad_hist else None
			eval_logs.append(eval_and_save(args, backbone, memory_loaders, test_loaders, savecount=task, use_cl_wrapper=use_cl_wrapper, 
				buf=buf, projector=projector, model=model.module))

			logs = dict(loss_ssl=loss_logs[0],
				w_ssl=w_logs[0],
				eval_logs=eval_logs)
			for li in range(len(loss_logs)-1):
				logs[f'loss_additional_{li}'] = loss_logs[li+1]
				logs[f'w_additional_{li}'] = w_logs[li+1]
			logs['seen_loss'] = seen_loss_logs
			logs['mem_loss'] = mem_loss_logs
			pickle.dump(logs, open(args.save_dir / 'logs.pkl', 'wb'))

			torch.distributed.barrier()
		else:
			torch.distributed.barrier()


	if args.rank == 0:
		torch.save(backbone.state_dict(), args.save_dir / 'resnet18.pth')

def eval_and_save(args, backbone, savecount, use_cl_wrapper, buf=None, buf_tl=None, projector=None, model=None):
	task_accs = []

	if use_cl_wrapper:
		torch.save(backbone.state_dict(), args.save_dir / ('backbone_'+str(savecount)+'.pth'))
	else:
		torch.save(backbone.state_dict(), args.save_dir / ('backbone_'+str(savecount)+'.pth'))

	if args.replay_mode == 'mixup' or args.grad_hist:
		torch.save(buf, args.save_dir / ('memory_'+str(savecount)+'.pt'))

	if ('sliding' in args.model and args.num_tasks > 1) or args.replay_mode == 'mixup':
		torch.save(projector.state_dict(), args.save_dir / ('projector_'+str(savecount)+'.pth'))

	if args.grad_hist:
		torch.save(model.state_dict(), args.save_dir / ('model_'+str(savecount)+'.pth'))

	return task_accs


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default='tinyimagenet', choices=['cifar100', 'tinyimagenet'])
	parser.add_argument('--num_classes', type=int, default=200)
	parser.add_argument('--num_tasks', type=int, default=10)
	parser.add_argument('--order', type=str, default='task_iid', 
						choices=['task_iid'])
	parser.add_argument('--model', type=str)

	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--warmup_epochs', type=int, default=None)
	parser.add_argument('--warmup_steps', type=int, default=0)

	parser.add_argument('--eval_freq', type=int, default=1955)
	parser.add_argument('--eval_batch_size', type=int, default=64)
	parser.add_argument('--knn_k', type=int, default=200)
	parser.add_argument('--knn_t', type=float, default=0.1)
	parser.add_argument('--data_dir', type=Path, metavar='DIR')

	parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'])
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--learning_rate_weights', type=float, default=0.03)
	parser.add_argument('--learning_rate_biases', type=float, default=0.03)
	parser.add_argument('--lr_decay', type=float, default=1)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--momentum', default=0.9, type=float)

	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency')
	parser.add_argument('--save_freq', default=1, type=int, metavar='N', help='save frequency')

	parser.add_argument('--projector', default='2048-128', type=str, metavar='MLP', help='projector MLP')
	parser.add_argument('--predictor', default='2048-128', type=str, metavar='MLP', help='predictor MLP')

	parser.add_argument('--order_dir', type=Path, metavar='DIR')
	parser.add_argument('--save_dir', type=Path, metavar='DIR')

	parser.add_argument("--distributed", action='store_true', help="multi gpu")
	parser.add_argument('--rank', default=0, type=int)
	parser.add_argument('--local_rank', type=int)
	parser.add_argument('--world_size', default=2, type=int, help='number of gpus')

	parser.add_argument('--seed', default=88, type=int)

	parser.add_argument('--norm', type=str, default='gn', choices=['bn', 'gn'])
	parser.add_argument('--act', type=str, default='mish', choices=['relu', 'mish'])
	parser.add_argument('--proj_design', type=str, default='empty_relu', choices=['bn_relu', 'gn_mish', 'empty_relu'])

	parser.add_argument('--buffer_size', type=int, default=256)
	parser.add_argument('--topk', type=int, default=-1)
	parser.add_argument('--replay_mode', type=str, default='concat')
	parser.add_argument('--p', type=float, default=0.5, help='prob of selecting current sample')
	parser.add_argument('--temp', type=float, default=0.1, help='softmax temperature')
	parser.add_argument('--alpha', type=float, default=0.4, help='mixup param')

	parser.add_argument('--gamma', default=1, type=float, help='weight for reservoir sampling')
	parser.add_argument('--w_simclr', default=1, type=float, help='weight for simclr loss')
	parser.add_argument('--w_additional', nargs='+', default=[], type=float, help='weight for additional losses')

	parser.add_argument('--memory_weighting', type=str, default='unif')
	parser.add_argument("--debug_overfit", action='store_true')
	parser.add_argument("--final_step", type=int, default=-1)
	parser.add_argument("--from_ckpt", action='store_true')
	parser.add_argument("--mixup", action='store_true')

	parser.add_argument('--min_size', type=float, default=0.08, help='min crop scale')

	parser.add_argument("--ewc_gamma", default=0.95, type=float)
	parser.add_argument("--max_lambda", default=0.8, type=float)
	parser.add_argument("--min_lambda", default=0.2, type=float)
	parser.add_argument("--ema_m", default=0.999, type=float)
	parser.add_argument("--pocon_ds", default=2000, type=int, help='pocon number of steps to update expert ckpt')

	parser.add_argument("--grad_hist", action='store_true')

	args = parser.parse_args()

	print(args)

	set_seed(args)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	with open(args.save_dir / 'args.txt', 'w') as f:
		args_copy = deepcopy(args.__dict__)
		args_copy['order_dir'] = str(args.order_dir)
		args_copy['save_dir'] = str(args.save_dir)
		args_copy['data_dir'] = str(args.data_dir)
		json.dump(args_copy, f, indent=2)

	if args.distributed:
		assert torch.distributed.is_available()
		print("PyTorch Distributed available.")
		print("  Backends:")
		print(f"    Gloo: {torch.distributed.is_gloo_available()}")
		print(f"    NCCL: {torch.distributed.is_nccl_available()}")
		print(f"    MPI:  {torch.distributed.is_mpi_available()}")

		# rank = int(os.environ["SLURM_PROCID"])
		# args.rank = rank

		world_size = int(os.environ["SLURM_NTASKS"])
		args.world_size = world_size
		assert args.world_size == torch.cuda.device_count()

		#os.environ["RANK"] = str(rank)
		#os.environ["WORLD_SIZE"] = str(world_size)

		torch.multiprocessing.spawn(main_worker, nprocs=args.world_size, args=(args,))
		#main_worker(args)

	else:
		pass


if __name__ == '__main__':
	main()