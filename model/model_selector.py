import os

import torch
from torch import nn
from torchvision.models import \
    resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, \
    ResNet50_Weights, resnet50, ResNet101_Weights, resnet101, \
    resnet152, ResNet152_Weights


def get_default_resnet(layers: int = 18, num_classes: int = 10, pretrain: bool = True) -> nn.Module:
    """
    get PyTorch default ResNet model

    :param layers: number of layers of model
    :param num_classes: output dimension
    :param pretrain: load parameters from online
    :return: pytorch nn Module
    """
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
    in_ftr = net.fc.in_features  # Input dimension of fully connected (lc) layer
    net.fc = nn.Linear(in_ftr, num_classes, bias=True)  # Output dimension
    return net


def get_checkpoint_resnet(load_path: str, layers: int = 18, num_classes: int = 10) -> nn.Module:
    net = get_default_resnet(layers, num_classes)

    device_name: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    augmented_path: str = get_or_create_checkpoint_path(load_path)
    checkpoint = torch.load(augmented_path, map_location=device_name)
    net = torch.nn.DataParallel(net)
    if device_name == 'cuda':
        torch.backends.cudnn.benchmark = True
    net.load_state_dict(checkpoint['state_dict'])

    return net


def get_or_create_checkpoint_path(input_path) -> str:
    augmented_path = os.path.join("checkpoint", input_path)
    checkpoint_dir: str = os.path.dirname(augmented_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return augmented_path
