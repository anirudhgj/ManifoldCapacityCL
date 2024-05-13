# %%
import numpy as np
import random
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision import datasets, transforms, models
import os

seed = 0
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
import sys
import torchvision

sys.path.append("../")
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze

# %%
from avalanche.training import EWC

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.models.pytorchcv_wrapper import resnet, vgg
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint

# %%

model = torchvision.models.resnet18(num_classes=100).to(device)

# CL Benchmark Creation
# perm_mnist = PermutedMNIST(n_experiences=1, seed=0)
split_CIFAR = SplitCIFAR100(n_experiences=10, seed=0, return_task_id=True)
train_stream = split_CIFAR.train_stream
test_stream = split_CIFAR.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()

# Continual learning strategy
cl_strategy = EWC(
    model,
    optimizer,
    criterion,
    ewc_lambda=1000,
    train_mb_size=2048,
    train_epochs=150,
    eval_mb_size=32,
    device=device,
)


project_name = "ewc1000_CIFAR100"
# cl_strategy, initial_exp = maybe_load_checkpoint(cl_strategy, "./0_checkpoint.pth")
if not os.path.exists(f"./results/{project_name}/"):
    os.makedirs(f"./results/{project_name}/")


# train and test loop over the stream of experiences
results = []
for c, train_exp in enumerate(train_stream):
    # cl_strategy.eval(test_stream)
    cl_strategy.train(train_exp)
    save_checkpoint(cl_strategy, f"./results/{project_name}/{c}_checkpoint.pth")
    results.append(cl_strategy.eval(test_stream))

# %%
