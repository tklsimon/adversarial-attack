from abc import ABC

import torchvision.transforms as transforms
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18, resnet50, ResNet50_Weights

from .abstract_baseline_model import AbstractBaselineModel


class ClfarDatsetModel(AbstractBaselineModel, ABC):
    def _set_train_set(self) -> Dataset:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return CIFAR10(root='./data', train=True, download=True, transform=transform)

    def _set_test_set(self) -> Dataset:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return CIFAR10(root='./data', train=False, download=True, transform=transform)


class Resnet18Model(ClfarDatsetModel):

    def _set_model(self) -> Module:
        return resnet18(weights=ResNet18_Weights.DEFAULT)


class Resnet50Model(ClfarDatsetModel):

    def _set_model(self) -> Module:
        return resnet50(weights=ResNet50_Weights.DEFAULT)
