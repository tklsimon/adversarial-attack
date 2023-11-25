import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import Dataset

from .attack_scenario import AttackScenario


class PgdAttackScenario(AttackScenario):
    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, test_val_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.03,
                 alpha: float = 0.007, noise_epochs: int = 10):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, test_val_ratio=test_val_ratio,
                         model=model, train_set=train_set, test_set=test_set)
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.noise_epochs: int = noise_epochs

    def __str__(self):
        return "Scenario=%s, model=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, weigh_decay=%.2E, momentum=%.2E, " \
               "test_val_ratio=%.2E, epsilon=%.2E, alpha=%.2E, num_iter=%d" % (
                   self.__class__.__name__,
                   self.model.__class__.__name__,
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.test_val_ratio, self.epsilon, self.alpha, self.noise_epochs)

    def attack(self, model: Module, inputs: Tensor, targets: Tensor) -> Tensor:
        return pgd_attack(model, inputs, targets, self.noise_epochs, self.epsilon, self.alpha)


def pgd_attack(model: Module, inputs: Tensor, targets: Tensor, noise_epochs: int, epsilon: float,
               alpha: float) -> Tensor:
    with torch.enable_grad():
        # initialize attack settings
        _inputs = inputs.clone().detach()
        _targets = targets.clone().detach()
        criterion = CrossEntropyLoss()

        perturbed_input = inputs.clone().detach()
        for i in range(noise_epochs):
            # enable grad for inputs
            perturbed_input.requires_grad = True

            # Forward pass the data through the model
            output = model(perturbed_input)

            # Calculate the loss
            loss: Tensor = criterion(output, _targets)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_input = perturbed_input.detach() + alpha * perturbed_input.grad.sign()
            # confining perturbation
            eta = torch.clamp(perturbed_input - _inputs, -epsilon, epsilon)
            # Adding clipping to maintain [0,1] range
            perturbed_input = torch.clamp(_inputs + eta, 0, 1)

        # Return the perturbed image
        return perturbed_input.detach()
