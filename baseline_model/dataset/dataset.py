from typing import Tuple

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def get_default_cifar10_dataset(is_train: bool, download: bool = False) -> Dataset:
    """
    Get raw CIFAR10 dataset

    :arg is_train: train set or test set
    :arg download: need to download file

    :return Pytorch Dataset: CIFAR10 dataset without any transformation
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return CIFAR10(root='./data', train=is_train, download=download, transform=transform)


def get_normalized_cifar10_dataset(is_train: bool, download: bool = False) -> Dataset:
    """
    Get normalized CIFAR10 images

    :param is_train: train set or test set
    :param download: need to download file
    :return Pytorch Dataset: CIFAR10 dataset with normalized transformation
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return CIFAR10(root='./data', train=is_train, download=download, transform=transform)


def get_random_cifar10_dataset(is_train: bool, download: bool = False) -> Dataset:
    """
    Get random processed CIFAR10 Dataset

    :param is_train: train set or test set
    :param download: need to download file
    :return Pytorch Dataset: CIFAR10 dataset with randomized transformation such as brightness or rotation
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    return CIFAR10(root='./data', train=is_train, download=download, transform=transform)


def get_cifar10_targets() -> Tuple:
    """
    Get the target classes of CIFAR10 dataset

    :return tuple containing target classes of CIFAR10 dataset
    :rtype tuple
    """
    return 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
