from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50

from .resnet import ResNet, Bottleneck, BasicBlock


def get_default_resnet18(num_classes: int = 10) -> nn.Module:
    net: nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(512, num_classes)
    return net


def get_default_resnet50(num_classes: int = 10) -> nn.Module:
    net: nn.Module = resnet50(weights=ResNet50_Weights.DEFAULT)
    net.fc = nn.Linear(512, num_classes)
    return net


def get_custom_resnet18(num_classes: int = 10) -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def get_custom_resnet34(num_classes: int = 10) -> nn.Module:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def get_custom_resnet50(num_classes: int = 10) -> nn.Module:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def get_custom_resnet101(num_classes: int = 10) -> nn.Module:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def get_custom_resnet152(num_classes: int = 10) -> nn.Module:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
