from torchvision.models import ResNet50_Weights, resnet50, ResNet

from .baseline_model import BaselineModel


class Resnet50Model(BaselineModel):

    def get_model(self) -> ResNet:
        return resnet50(weights=ResNet50_Weights.DEFAULT)
