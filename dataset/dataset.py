from typing import Tuple, List

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from dataset.custom_transform import MaskTransform


def get_cifar10_targets() -> Tuple:
    """
    Get the target classes of CIFAR10 dataset

    :return tuple containing target classes of CIFAR10 dataset
    :rtype tuple
    """
    return 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'


def get_cifar10_dataset(is_train: bool, download: bool = False, transform: str = '') -> Dataset:
    """

    :arg is_train: train set or test set
    :arg download: need to download file
    :param transform: name of transformation
    :return:
    """
    if transform == 'random':
        transform = get_random_transform()
    elif transform == 'normalize':
        transform = get_normalize_transform()
    elif transform == 'dropout':
        transform = get_dropout_transform()
    elif transform == 'random-blur':
        transform = get_random_blur_transform()
    elif transform == 'mask':
        transform = get_random_blur_transform()
    else:
        transform = get_default_transform()
    return CIFAR10(root='data', train=is_train, download=download, transform=transforms.Compose(transform))


def get_default_transform() -> List:
    return [
        transforms.ToTensor()
    ]


def get_random_transform() -> List:
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ]


def get_normalize_transform() -> List:
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]


def get_dropout_transform() -> List:
    return [
        transforms.ToTensor(),
        MaskTransform(0.3)
    ]


def get_random_blur_transform() -> List:
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=1.0),
        transforms.ToTensor()
    ]
