import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .train_test_scenario import TrainTestScenario
from ..dataset.fgsm_attack import fgsm_attack


class FgsmAttackScenario(TrainTestScenario):
    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, train_eval_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.07):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, train_eval_ratio=train_eval_ratio,
                         model=model, train_set=train_set, test_set=test_set)
        self.epsilon = epsilon

    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: _Loss) -> float:
        model.eval()  # switch to evaluation mode
        loss_value = 0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress_bar:
                inputs: Tensor = inputs.to(device_name)
                targets: Tensor = targets.to(device_name)
                inputs.requires_grad = True

                outputs = model(inputs)

                loss = F.nll_loss(outputs, targets)
                model.zero_grad()
                loss.backward()
                perturbed_data_normalized = fgsm_attack(inputs, self.epsilon)
                outputs = model(perturbed_data_normalized)

                loss_value += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    loss_value / (batch_idx + 1), 100. * correct / total, correct, total
                )
                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return loss_value / len(data_loader)
