import torch
import torch
from torchvision import datasets, transforms, models
import sys
import numpy as np

sys.path.append("../../")
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze


from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_dataloaders():

    test_transform = transforms.Compose(
        [
            transforms.Resize(32, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    trainset = datasets.CIFAR100(
        root="./", train=True, download=True, transform=test_transform
    )

    testset = datasets.CIFAR100(
        root="./", train=False, download=True, transform=test_transform
    )

    test_loader = DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    return (
        test_loader,
        trainset,
        test_loader,
        testset,
    )


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

    return capacities, radii, dimensions, correlations, names
