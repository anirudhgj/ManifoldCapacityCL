#####
# Built upon https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
#####

import torch.nn.functional as F 
import torch
import numpy as np

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k=200, knn_t=0.1):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels, pred_scores 

# test without task labels
def knn_test(args, net, memory_data_loader, test_data_loader, num_tasks, class_order=None):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    feature_bank = []
    all_pred, all_gt = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels, _ = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            all_pred += pred_labels[:, 0].tolist()
            all_gt += target.tolist()

    all_pred = np.array(all_pred).astype(int)
    all_gt = np.array(all_gt).astype(int)

    classes_per_task = classes // num_tasks
    task_accs = []
    class_order = list(range(classes)) if class_order is None else class_order


    for i in range(num_tasks):
        curr_classes = class_order[classes_per_task*i:classes_per_task*i+classes_per_task]
        curr_pred, curr_gt = np.zeros(0), np.zeros(0)
        for c in curr_classes:
            idx = (all_gt == c)
            curr_pred = np.concatenate([curr_pred, all_pred[idx]])
            curr_gt = np.concatenate([curr_gt, all_gt[idx]])
        acc = (curr_pred == curr_gt).sum() / curr_gt.shape[0]
        task_accs.append(float(acc))

    return task_accs


# test with task labels
def knn_test_wtl(args, net, memory_data_loader, test_data_loader):
    net.eval()
    classes = len(memory_data_loader.dataset.classes) 
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) # these are remapped to {0, 1, ..., num_class_per_task-1}
        
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels, _ = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return (total_top1 / total_num)


# test on the task level
def knn_test_task(args, net, memory_data_loader, test_data_loader, num_tasks=20, class_order=None):
    net.eval()

    classes = len(memory_data_loader.dataset.classes)
    class_order = list(range(classes)) if class_order is None else class_order
    classes_per_task = classes // num_tasks
    label2task = np.zeros(classes)
    for t in range(num_tasks):
        for i in range(classes_per_task):
            c = class_order[classes_per_task*t+i]
            label2task[c] = t

    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        targets = [label2task[c] for c in memory_data_loader.dataset.targets]
        feature_labels = torch.tensor(targets, dtype=torch.long, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        pred = torch.zeros((0, num_tasks), device=feature_bank.device)
        gt = []
        for data, target in test_data_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            _, pred_scores = knn_predict(feature, feature_bank, feature_labels, num_tasks, args.knn_k, args.knn_t)

            pred = torch.cat([pred, pred_scores], dim=0)
            gt += [label2task[c] for c in target.tolist()]

    pred = pred.argsort(dim=-1, descending=True)[:, 0].detach().cpu().numpy()
    gt = np.array(gt).astype(int)
    acc = (pred == gt).sum() / gt.shape[0]

    return acc