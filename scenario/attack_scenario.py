from typing import Dict

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_scenario import BaseScenario


class AttackScenario(BaseScenario):
    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: Module) -> Dict:
        model.eval()  # switch to evaluation mode
        loss_value = 0
        correct = 0
        total = 0
        similarities = 0
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
                similarities += similarity(inputs, perturbed_input)

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    loss_value / (batch_idx + 1), 100. * correct / total, correct, total
                )
                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return {'average test_loss': loss_value / len(data_loader), 'accuracy': correct / total,
                'average similarity': similarities / total}

    def attack(self, model: Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        transform input to noise image

        :param model: model to be attack
        :param inputs: original input
        :param targets: original target
        :return: input with noise
        """
        return inputs.detach()


def similarity(original_images: torch.Tensor, adversarial_images: torch.Tensor):
    # Move images back to CPU for visualization
    original_images = original_images.cpu().numpy()
    adversarial_images = adversarial_images.detach().cpu().numpy()

    # Compute cosine similarity
    images_flatten = original_images.reshape(original_images.shape[0], -1)
    adversarial_images_flatten = adversarial_images.reshape(adversarial_images.shape[0], -1)

    cosine_similarities = F.cosine_similarity(torch.from_numpy(images_flatten),
                                            torch.from_numpy(adversarial_images_flatten))
    return cosine_similarities.sum().item()
