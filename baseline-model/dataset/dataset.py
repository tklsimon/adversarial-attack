from typing import Tuple

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def get_default_cifar10_dataset(download: bool = False) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return CIFAR10(root='./data', train=True, download=download, transform=transform), \
           CIFAR10(root='./data', train=False, download=download, transform=transform)


def get_normalized_cifar10_dataset(download: bool = False) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return CIFAR10(root='./data', train=True, download=download, transform=transform), \
           CIFAR10(root='./data', train=False, download=download, transform=transform)


def get_random_cifar10_dataset(download: bool = False) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return CIFAR10(root='./data', train=True, download=download, transform=transform), \
           CIFAR10(root='./data', train=False, download=download, transform=test_transform)
