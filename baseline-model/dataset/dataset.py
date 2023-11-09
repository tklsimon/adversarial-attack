import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from typing import Tuple


def get_default_cifar10_dataset() -> Tuple[Dataset, Dataset]:
    return CIFAR10(root='./data', train=True, download=True), CIFAR10(root='./data', train=False, download=True)


def get_normalized_cifar10_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return CIFAR10(root='./data', train=True, download=True, transform=transform), \
           CIFAR10(root='./data', train=False, download=True, transform=transform)
