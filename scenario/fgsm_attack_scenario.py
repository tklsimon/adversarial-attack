import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .base_scenario import BaseScenario


class FgsmAttackScenario(BaseScenario):
    """
    Base implementation for Scenario, only contains basic training function for a baseline model
    """
    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, test_val_ratio: float = 0.5,
                 model: nn.Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.07):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, test_val_ratio=test_val_ratio,
                         model=model, train_set=train_set, test_set=test_set)
        self.epsilon: float = epsilon

    def __str__(self):
        return "Scenario=%s, model=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, weigh_decay=%.2E, momentum=%.2E, " \
               "test_val_ratio=%.2E, epsilon=%d" % (
                   self.__class__.__name__,
                   self.model.__class__.__name__,
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.test_val_ratio, self.epsilon)

    def attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return fgsm_attack(model, inputs, targets, self.epsilon)


def fgsm_attack(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    with torch.enable_grad():
        # initialize attack settings
        _inputs = inputs.clone().detach()
        _targets = targets.clone().detach()
        criterion = nn.CrossEntropyLoss()

        # enable grad for inputs
        _inputs.requires_grad = True

        # Forward pass the data through the model
        output = model(_inputs)

        # Calculate the loss
        loss: torch.Tensor = criterion(output, _targets)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_input = _inputs.detach() + epsilon * _inputs.grad.sign()
        # Adding clipping to maintain [0,1] range
        perturbed_input = torch.clamp(perturbed_input, 0, 1)
        # Return the perturbed image
        return perturbed_input
