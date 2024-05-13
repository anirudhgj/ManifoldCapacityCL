# %%
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import utils
import glob
from natsort import natsorted

import sys
import torchvision

sys.path.append("../../")
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze

from data import PermutedMNIST
from utils import EWC, ewc_train, normal_train, test


seed = 0
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


epochs = 50
lr = 1e-3
batch_size = 128
sample_size = 200
hidden_size = 200
num_task = 3

project_name = "naive_MNIST"
weight_files = natsorted(glob.glob(f"./results/{project_name}/*.pth"))

# %%


def get_permute_mnist():
    train_loader = {}
    test_loader = {}
    idx = list(range(28 * 28))
    for i in range(num_task):
        train_loader[i] = torch.utils.data.DataLoader(
            PermutedMNIST(train=True, permute_idx=idx),
            batch_size=batch_size,
            num_workers=4,
        )
        test_loader[i] = torch.utils.data.DataLoader(
            PermutedMNIST(train=False, permute_idx=idx), batch_size=batch_size
        )
        random.shuffle(idx)
    return train_loader, test_loader


train_loader, test_loader = get_permute_mnist()


# %%
class MLP(nn.Module):
    def __init__(self, hidden_size=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


# %%
# Config

metrics = {}

for i in range(len(weight_files)):
    print(i, weight_files[i])

    model = MLP(hidden_size).to(device)

    model.load_state_dict(torch.load(weight_files[i]))

    print("Model loaded")
    data = []
    for i in range(1):
        task_data = []
        for input, target in train_loader[i]:
            input, target = utils.variable(input), utils.variable(target)
            data.append(input.detach().cpu().numpy())
        # data.append(task_data)
    data = torch.Tensor(data[:5]).to(device)
    print("Data created")

    activations = extractor(model, data, layer_types=["Linear"])
    # capacities, radii, dimensions, correlations, names = run_utils.get_manifold_metrics(
    #     model
    # )
    print("Extracted activations")

    for layer, data in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]
        activations[layer] = X

    print("calculating metrics")
    capacities = []
    radii = []
    dimensions = []
    correlations = []
    names = []

    cnt = 0

    for k, X in activations.items():
        cnt += 1
        # Analyze each layer's activations
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 50, n_reps=1)

        # Compute the mean values
        a = 1 / np.mean(1 / a)
        r = np.mean(r)
        d = np.mean(d)
        print(
            "{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(
                k, a, r, d, r0
            )
        )

        # Store for later
        capacities.append(a)
        radii.append(r)
        dimensions.append(d)
        correlations.append(r0)
        names.append(k)

    metrics.update(
        {
            f"task_{i}": {
                "capacities": capacities,
                "radii": radii,
                "dimensions": dimensions,
                "correlations": correlations,
                "names": names,
            }
        }
    )

    if not os.path.exists(f"./plots/{project_name}"):
        os.makedirs(f"./plots/{project_name}")

    torch.save(metrics, f"./plots/{project_name}/naive_CIFAR100_metrics.pth")
