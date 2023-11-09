from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50

from .resnet import ResNet, ResidualBlock


def get_default_resnet18() -> nn.Module:
    net: nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(512, 10)
    return net


def get_default_resnet50() -> nn.Module:
    net: nn.Module = resnet50(weights=ResNet50_Weights.DEFAULT)
    net.fc = nn.Linear(512, 10)
    return net


def get_custom_resnet() -> nn.Module:
    return ResNet(ResidualBlock, [2, 2, 2])
