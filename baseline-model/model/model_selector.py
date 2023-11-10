from torch import nn
from torch.utils import model_zoo
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50, ResNet101_Weights, resnet101, \
    resnet152, ResNet152_Weights, ResNet34_Weights, resnet34

from .resnet import ResNet, Bottleneck, BasicBlock


def get_default_resnet(layers: int = 18, num_classes: int = 10, pretrain: bool = True) -> nn.Module:
    net: nn.Module = None
    if layers == 18:
        weights = ResNet18_Weights.DEFAULT if pretrain else None
        net = resnet18(weights=weights)
    if layers == 34:
        weights = ResNet34_Weights.DEFAULT if pretrain else None
        net = resnet34(weights=weights)
    if layers == 50:
        weights = ResNet50_Weights.DEFAULT if pretrain else None
        net = resnet50(weights=weights)
    if layers == 101:
        weights = ResNet101_Weights.DEFAULT if pretrain else None
        net = resnet101(weights=weights)
    if layers == 152:
        weights = ResNet152_Weights.DEFAULT if pretrain else None
        net = resnet152(weights=weights)
    net.fc = nn.Linear(512, num_classes)
    return net


def get_custom_resnet(layers: int = 18, num_classes: int = 10) -> nn.Module:
    net: nn.Module = None
    if layers == 18:
        net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    if layers == 34:
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    if layers == 50:
        net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    if layers == 101:
        net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    if layers == 152:
        net = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    return net
