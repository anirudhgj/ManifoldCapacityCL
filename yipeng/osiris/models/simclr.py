import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ddp_utils import GatherLayer
from models.backbone import resnet18


def _make_projector(sizes):
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        layers.append(nn.ReLU(inplace=True))
    
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

    return nn.Sequential(*layers)


def _mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask.fill_diagonal_(0)
    mask[:, batch_size:].fill_diagonal_(0)
    mask[batch_size:, :].fill_diagonal_(0)
    return mask


class NT_Xent(nn.Module):
    """
    https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
    """
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = _mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, z_i, z_j):
        """
        Standard full contrastive loss on z
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # https://github.com/Spijkervet/SimCLR/issues/37
        z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j = torch.cat(GatherLayer.apply(z_j), dim=0)

        batch_size = z_i.size(0)
        N = 2 * batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        sim = z @ z.t() / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = _mask_correlated_samples(batch_size) if batch_size != self.batch_size else self.mask
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    

class Cross_NT_Xent(nn.Module):
    """
    https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
    """
    def __init__(self, batch_size, temperature):
        super(Cross_NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.batch_size = batch_size
        self.mask = _mask_correlated_samples(batch_size)

    def forward(self, z_i, z_j, u_i, u_j):
        """
        Contrastive loss for discriminating z and u
        No self comparison within z or u
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        u_i = F.normalize(u_i, p=2, dim=1)
        u_j = F.normalize(u_j, p=2, dim=1)

        # https://github.com/Spijkervet/SimCLR/issues/37
        z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        u_i = torch.cat(GatherLayer.apply(u_i), dim=0)
        u_j = torch.cat(GatherLayer.apply(u_j), dim=0)

        batch_size = z_i.size(0)
        N = batch_size * 2

        # current
        z = torch.cat([z_i, z_j], dim=0)
        sim_c = z @ z.t()
        sim_c_ij = torch.diag(sim_c, batch_size)
        sim_c_ji = torch.diag(sim_c, -batch_size)
        pos_c = torch.cat([sim_c_ij, sim_c_ji], dim=0).reshape(N, 1)

        # memory
        u = torch.cat([u_i, u_j], dim=0)
        neg_cm = z @ u.t()

        # loss
        labels = torch.zeros(N).to(pos_c.device).long()
        logits = torch.cat([pos_c, neg_cm], dim=1) / self.temperature
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    

class Distill_NT_Xent(nn.Module):
    """
    https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
    """
    def __init__(self, batch_size, temperature):
        super(Distill_NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = _mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        Contrastive loss for using z_i to predict data from z_j
        No self comparison within z_i or z_j
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # https://github.com/Spijkervet/SimCLR/issues/37
        z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j = torch.cat(GatherLayer.apply(z_j), dim=0)

        batch_size = z_i.size(0) // 2
        N = batch_size * 2

        sim = z_i @ z_j.t() / self.temperature

        positive_samples = torch.diag(sim).reshape(N, 1)

        mask = _mask_correlated_samples(batch_size) if batch_size != self.batch_size else self.mask
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    

class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(zero_init_residual=True, norm=args.norm, act=args.act)
        self.backbone.fc = nn.Identity()
        self.criterion = NT_Xent(args.batch_size, temperature=args.temp)

        sizes = [512] + list(map(int, args.projector.split('-')))
        self.projector = _make_projector(sizes)

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        loss = self.criterion(z1, z2)
        loss = [loss, [torch.zeros_like(loss)]]

        return loss
    

class Osiris(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(zero_init_residual=True, norm=args.norm, act=args.act)
        self.backbone.fc = nn.Identity()

        batch_size_m = int(args.batch_size // args.world_size * (1 - args.p)) * 4
        batch_size_c = args.batch_size

        self.criterion_curr = NT_Xent(batch_size_c, temperature=args.temp)
        self.criterion_cross = Cross_NT_Xent(batch_size_c, temperature=args.temp)

        self.criterion_replay = NT_Xent(batch_size_m, temperature=args.temp)
        self.criterion_distill = Distill_NT_Xent(batch_size_c, temperature=args.temp)

        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        self.projector = _make_projector(sizes)

        # predictor
        sizes = [512] + list(map(int, args.predictor.split('-')))
        self.predictor = _make_projector(sizes)
        for param in self.predictor.parameters():
            param.requires_grad = False


    def forward(self, xy1, xy2, z1_prev=None, z2_prev=None, mem_idx=None, task=None):
        
        zu1 = self.backbone(xy1)
        zu2 = self.backbone(xy2)

        if task == 0:
            # the first task
            z1 = self.projector(zu1)
            z2 = self.projector(zu2)
            loss = self.criterion_curr(z1, z2)
            loss = [loss, [torch.zeros_like(loss)]]

        else:
            z1, z2 = zu1[~mem_idx], zu2[~mem_idx]
            u1, u2 = zu1[mem_idx], zu2[mem_idx]

            # space 1
            z1_s1 = self.projector(z1)
            z2_s1 = self.projector(z2)
            loss1 = self.criterion_curr(z1_s1, z2_s1)

            # space 2
            z1_s2 = self.predictor(z1)
            z2_s2 = self.predictor(z2)
            u1_s2 = self.predictor(u1)
            u2_s2 = self.predictor(u2)
            loss2 = (self.criterion_cross(z1_s2, z2_s2, u1_s2, u2_s2) \
                     + self.criterion_cross(u1_s2, u2_s2, z1_s2, z2_s2)) / 2

            # also space 2
            if self.args.model == 'osiris-d':
                z1_prev = self.predictor(z1_prev)
                z2_prev = self.predictor(z2_prev)
                z = torch.cat([z1_prev, z2_prev], dim=0)
                p = torch.cat([z1_s2, z2_s2], dim=0)
                loss3 = (self.criterion_distill(p, z) \
                        + self.criterion_distill(z, p)) / 2
            
            elif self.args.model == 'osiris-r':
                loss3, _ = self.criterion_replay(u1_s2, u2_s2)

            else:
                raise NotImplementedError

            loss = [loss1, [loss2, loss3]]

        return loss







