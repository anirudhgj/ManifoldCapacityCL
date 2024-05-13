import torch.nn as nn
import torch
import copy

from models.buffer import Buffer
from utils.ddp_utils import concat_all_gather

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    def __init__(self, args, net):
        super(ContinualModel, self).__init__()

        self.net = net
        self.net_prev = None
        self.args = args
        self.buffer = Buffer(args)
        self.mode = args.replay_mode

    def forward(self, aug_x1, aug_x2, img, labels, task=None):
        x1, x2, z1_prev, z2_prev, mem_idx, task_labels = self.recall(aug_x1, aug_x2, img, labels, task)

        if task > 0 and self.args.model == 'osiris-d':
            z1_prev = self.net_prev.backbone(aug_x1)
            z2_prev = self.net_prev.backbone(aug_x2)

        if not z1_prev is None:
            z1_prev = z1_prev.detach()
            z2_prev = z2_prev.detach()

        loss = self.net(x1, x2, z1_prev, z2_prev, mem_idx, task, task_labels)
        img = concat_all_gather(img)
        self.buffer.add_data(img, task)

        return loss

    @torch.no_grad()
    def recall(self, x1, x2, task):
        if task > 0:
            assert self.buffer.num_seen_examples >= self.args.batch_size
            
            p = self.args.p
            per_gpu_k = int(x1.size(0) * (1-p)) 
            k = self.args.world_size * per_gpu_k
            start = self.args.rank * per_gpu_k

            buf_x1, buf_x2, z1_prev, z2_prev, task_labels = self.buffer.get_data(k, segment=[start, start+per_gpu_k], task=task)
            mixed_x1, mixed_x2, mem_idx, task_labels = self.select(x1, x2, buf_x1, buf_x2, task_labels, p, task)

        else:
            mixed_x1, mixed_x2 = x1, x2
            mem_idx = None
            task_labels = None
            z1_prev, z2_prev = None, None

        return mixed_x1, mixed_x2, z1_prev, z2_prev, mem_idx, task_labels

    @torch.no_grad()
    def select(self, x1, x2, buf_x1, buf_x2, task_labels, p, task):
        mem_idx = None
        
        mixed_x1 = torch.cat([x1, buf_x1], dim=0)
        mixed_x2 = torch.cat([x2, buf_x2], dim=0)
        mem_idx = torch.zeros(x1.size(0)+buf_x1.size(0), dtype=torch.bool)
        mem_idx[x1.size(0):] = True

        curr_task_labels = torch.ones(x1.shape[0], dtype=int, device='cuda') * task
        task_labels = torch.cat([curr_task_labels, task_labels], dim=0)

        return mixed_x1, mixed_x2, mem_idx, task_labels

    def update_model_states(self, task):
        if task == 0: 
            return

        if task == 1:
            if int(self.args.predictor.split('-')[0]) == int(self.args.projector.split('-')[0]):
                self.net.predictor.load_state_dict(self.net.projector.state_dict())
            for param in self.net.predictor.parameters():
                param.requires_grad = True

        if task > 0 and self.args.model == 'osiris-d':
            self.net_prev = copy.deepcopy(self.net)
            for param in self.net_prev.parameters():
                param.requires_grad = False

        return
