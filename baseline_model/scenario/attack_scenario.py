import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_model.scenario.base_scenario import BaseScenario


class AttackScenario(BaseScenario):
    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: _Loss) -> float:
        model.eval()  # switch to evaluation mode
        loss_value = 0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress_bar:
                inputs = inputs.to(device_name)
                targets = targets.to(device_name)
                perturbed_input = self.attack(model, inputs, targets)
                outputs = model(perturbed_input)
                loss = criterion(outputs, targets)

                loss_value += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    loss_value / (batch_idx + 1), 100. * correct / total, correct, total
                )
                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return loss_value / len(data_loader)

    def attack(self, model: Module, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        transform input to noise image

        :param model: model to be attack
        :param inputs: original input
        :param targets: original target
        :return: input with noise
        """
        return inputs.detach()
