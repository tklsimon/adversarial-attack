from torchvision.models import ResNet18_Weights, resnet18, ResNet

from .baseline_model import BaselineModel


class Resnet18Model(BaselineModel):

    def get_model(self) -> ResNet:
        return resnet18(weights=ResNet18_Weights.DEFAULT)