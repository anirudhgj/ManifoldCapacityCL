import torch
import numpy as np
from utils.loading_utils import BufferSSLTransform


def reservoir(num_seen_examples, buffer_size):
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples+1) # np sampling is exlusive on the upper bound
    return (rand if rand < buffer_size else -1)


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size
        self.feature_size = 128
        self.gamma = args.gamma
        self.num_seen_examples = 0
        self.num_seen_steps = 1
        self.rank = args.rank

        if args.dataset == 'cifar100':
            self.transform = BufferSSLTransform(min_size=args.min_size, in_size=32, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        elif args.dataset == 'tinyimagenet':
            self.transform = BufferSSLTransform(min_size=args.min_size, in_size=64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    @torch.no_grad()
    def add_data(self, examples, task):

        if not hasattr(self, 'examples'):
            self.examples = torch.zeros((self.buffer_size, *examples.shape[1:]), dtype=torch.float32, requires_grad=False).cpu()
            self.task_labels = torch.zeros(self.buffer_size, dtype=int)
            
        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size, self.gamma)
            if index >= 0:
                self.examples[index] = examples[i].cpu()
                self.task_labels[index] = task
            self.num_seen_examples += 1

        self.num_seen_steps += 1

    @torch.no_grad()
    def get_data(self, size, segment, task, exclude_current=False):
        # assume memory is fully filled during task 0
        if task > 0 and exclude_current:
            idx = (self.task_labels < task) 
            filtered_examples = self.examples[idx]
            filtered_tls = self.task_labels[idx]
            
        else:
            filtered_examples = self.examples
            filtered_tls = self.task_labels

        assert size <= filtered_examples.shape[0]
        choice = np.random.choice(filtered_examples.shape[0], size=size, replace=False)

        selected_x1, selected_x2 = [], []
        for ex in filtered_examples[choice][segment[0]:segment[1]]:
            ex1, ex2 = self.transform(ex)
            selected_x1.append(ex1)
            selected_x2.append(ex2)
        selected_x1 = torch.stack(selected_x1).cuda(non_blocking=True)
        selected_x2 = torch.stack(selected_x2).cuda(non_blocking=True)

        return selected_x1, selected_x2,    \
            filtered_tls[choice][segment[0]:segment[1]].cuda(non_blocking=True)
