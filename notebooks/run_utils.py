import torch
import torch
from torchvision import datasets, transforms, models
import sys
import numpy as np

sys.path.append("../")
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze


def get_dataloaders():
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR100(
        "../data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=True
    )

    return train_loader, train_dataset, test_loader, test_dataset


def create_manifold_data(
    train_dataset, device, sampled_classes=100, examples_per_class=50
):

    data = make_manifold_data(
        train_dataset, sampled_classes, examples_per_class, seed=0
    )
    data = [d.to(device) for d in data]

    return data


def get_manifold_metrics(model):

    train_loader, train_dataset, test_loader, test_dataset = get_dataloaders()

    data = create_manifold_data(
        train_dataset, torch.device("cuda:0"), 100, 50
    )  ## change this based on data

    model.eval()

    activations = extractor(model, data, layer_types=["Conv2d", "Linear"])

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

    capacities = []
    radii = []
    dimensions = []
    correlations = []
    names = []

    cnt = 0

    for k, X in activations.items():
        cnt += 1
        # Analyze each layer's activations
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)

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

        if cnt == 5:
            break

    return capacities, radii, dimensions, correlations, names
