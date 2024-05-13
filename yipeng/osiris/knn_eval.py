import os
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

from torch import nn
import torch

from models.backbone import resnet18

from utils.loading_utils import get_knn_data_loaders
from utils.knn_utils import knn_test, knn_test_task, knn_test_wtl



def prepare_model(args):
    model = resnet18(norm=args.norm)

    model.fc = nn.Identity()
    ckpt = torch.load(args.ckpt_dir / 'resnet18.pth')
    model.load_state_dict(ckpt)

    model.fc = nn.Identity()

    return model

def eval_w_hist(args, save_order=False):
    accs = []
    memory_loaders, test_loaders = get_knn_data_loaders(args)

    # initial eval
    model = resnet18(pretrained=False, norm=args.norm, act=args.act)
    model.fc = nn.Identity()
    if 'joint' in str(args.ckpt_dir):
        ckpt = torch.load(args.ckpt_dir / f'resnet18.pth')
        model.load_state_dict(ckpt)
    model.to('cuda:0')
    model.eval()
    
    task_accs = []
    for memory_loader, test_loader in zip(memory_loaders, test_loaders):
        acc = knn_test_wtl(args, model, memory_data_loader=memory_loader, test_data_loader=test_loader)
        task_accs.append(acc)
    accs.append(task_accs)
    
    num_ckpt = args.num_tasks
    for i in tqdm(range(num_ckpt), total=num_ckpt, desc='KNN Testing'):
        if 'joint' in str(args.ckpt_dir):
            break
            
        if not (args.ckpt_dir / f'backbone_{i}.pth').is_file():
            break
        model = resnet18(pretrained=False, norm=args.norm, act=args.act)
        model.fc = nn.Identity()
        ckpt = torch.load(args.ckpt_dir / f'backbone_{i}.pth')
        model.load_state_dict(ckpt)
        model.to('cuda:0')
        model.eval()

        task_accs = []
        for memory_loader, test_loader in zip(memory_loaders, test_loaders):
            acc = knn_test_wtl(args, model, memory_data_loader=memory_loader, test_data_loader=test_loader)
            task_accs.append(acc)
        accs.append(task_accs)

    order_dir = '' if not save_order else str(args.order_dir)[-3:]
    with open(args.ckpt_dir / f'eval_logs_wtl_{args.limit_visibility}{order_dir}.txt', 'w') as f:
        f.write('\n'.join([','.join([str(task_acc) for task_acc in el]) for el in accs]))



def eval_wo_hist(args, save_order=False):
    accs = []
    train_loader, test_loader = get_knn_data_loaders(args)

    class_to_idx = train_loader.dataset.class_to_idx
    class_order = pickle.load(open(args.order_dir / 'class_order.pkl', 'rb'))
    class_order = [class_to_idx[nid] for nid in class_order]

    # initial eval
    model = resnet18(pretrained=False, norm=args.norm, act=args.act)
    model.fc = nn.Identity()
    if 'joint' in str(args.ckpt_dir):
        ckpt = torch.load(args.ckpt_dir / f'resnet18.pth')
        model.load_state_dict(ckpt)
    model.to('cuda:0')
    model.eval()

    if args.limit_visibility == 'na':
        task_accs = knn_test(args, model, memory_data_loader=train_loader, test_data_loader=test_loader, 
            num_tasks=args.num_tasks, class_order=class_order)
        
    accs.append(task_accs)
    
    num_ckpt = args.num_tasks
    for i in tqdm(range(num_ckpt)):
        if 'joint' in str(args.ckpt_dir):
            break
            
        if not (args.ckpt_dir / f'backbone_{i}.pth').is_file():
            break
        model = resnet18(pretrained=False, norm=args.norm, act=args.act)
        model.fc = nn.Identity()
        ckpt = torch.load(args.ckpt_dir / f'backbone_{i}.pth')
        model.load_state_dict(ckpt)
        model.to('cuda:0')
        model.eval()

        if args.limit_visibility == 'na':
            task_accs = knn_test(args, model, memory_data_loader=train_loader, test_data_loader=test_loader, 
                num_tasks=args.num_tasks, class_order=class_order)

        accs.append(task_accs)

    order_dir = '' if not save_order else str(args.order_dir)[-3:]
    with open(args.ckpt_dir / f'eval_logs_wotl_{args.limit_visibility}{order_dir}.txt', 'w') as f:
        f.write('\n'.join([','.join([str(task_acc) for task_acc in el]) for el in accs]))


def task_level_knn(args):
    accs = []
    train_loader, test_loader = get_knn_data_loaders(args)

    class_to_idx = train_loader.dataset.class_to_idx
    class_order = pickle.load(open(args.order_dir / 'class_order.pkl', 'rb'))
    class_order = [class_to_idx[nid] for nid in class_order]

    # initial eval
    model = resnet18(pretrained=False, norm=args.norm, act=args.act)
    model.fc = nn.Identity()
    model.to('cuda:0')
    model.eval()

    if args.limit_visibility == 'na':
        acc = knn_test_task(args, model, memory_data_loader=train_loader, test_data_loader=test_loader, 
            num_tasks=args.num_tasks, class_order=class_order)

    accs.append(acc)
    
    for i in tqdm(range(args.num_tasks)):
        if 'joint' in str(args.ckpt_dir) and i > 0:
            break
        if not (args.ckpt_dir / f'backbone_{i}.pth').is_file():
            break

        model = resnet18(pretrained=False, norm=args.norm)
        model.fc = nn.Identity()

        if 'joint' in str(args.ckpt_dir):
            ckpt = torch.load(args.ckpt_dir / f'resnet18.pth')
            model.load_state_dict(ckpt)
        else:
            ckpt = torch.load(args.ckpt_dir / f'backbone_{i}.pth')

        model.load_state_dict(ckpt)
        model.to('cuda:0')
        model.eval()

        if args.limit_visibility == 'na':
            acc = knn_test_task(args, model, memory_data_loader=train_loader, test_data_loader=test_loader, 
                num_tasks=args.num_tasks, class_order=class_order)

        accs.append(acc)

    results = {'acc': accs}
    pickle.dump(results, open(args.ckpt_dir / f'task_knn_{args.limit_visibility}.pkl', 'wb'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='tinyimagenet', choices=['cifar100', 'tinyimagenet'])
    parser.add_argument('--order', type=str, default='iid')
    parser.add_argument('--data_dir', type=Path, metavar='DIR')
    parser.add_argument('--ckpt_dir', type=Path, metavar='DIR')

    parser.add_argument('--order_dir', type=Path, metavar='DIR')

    parser.add_argument('--pretrained_num_classes', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--num_tasks', type=int, default=20)

    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--knn_k', type=int, default=200)
    parser.add_argument('--knn_t', type=float, default=0.1)

    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--mode', type=str)

    parser.add_argument('--norm', type=str, choices=['bn', 'gn'], default='gn')
    parser.add_argument('--act', type=str, choices=['mish', 'relu'], default='mish')

    parser.add_argument('--limit_visibility', type=str, default='na')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.ckpt_dir):
        raise Exception('Incorrect directory.')

    if args.mode == 'mt_w_hist':
        print(str(args.order_dir)[-3:])
        save_order = True if args.order == 'iid' else False
        args.order = 'task_iid'
        eval_w_hist(args, save_order)

    elif args.mode == 'mt_wo_hist':
        print(str(args.order_dir)[-3:])
        save_order = True if args.order == 'iid' else False
        args.order = 'iid'
        eval_wo_hist(args, save_order)

    elif args.mode == 'task_knn':
        args.order = 'iid'
        task_level_knn(args)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()