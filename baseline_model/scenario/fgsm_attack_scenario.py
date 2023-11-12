import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from .train_test_scenario import TrainTestScenario


# from baseline_model.dataset.fgsm_attack import fgsm_attack

class FgsmAttackScenario(TrainTestScenario):
    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, train_eval_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.07):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, train_eval_ratio=train_eval_ratio,
                         model=model, train_set=train_set, test_set=test_set)
        self.epsilon = epsilon

    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: _Loss) -> float:
        epsilon = self.epsilon
        model.eval()  # switch to evaluation mode
        loss_value = 0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        # Loop over all examples in test set
        for batch_idx, (inputs, targets) in progress_bar:
            # Send the data and label to the device
            inputs = inputs.to(device_name)
            targets = targets.to(device_name)

            # Set requires_grad attribute of tensor. Important for Attack
            inputs.requires_grad = True

            # Forward pass the data through the model
            output = model(inputs)

            # Calculate the loss
            loss = F.nll_loss(output, targets)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # get FGSM noise inputs
            perturbed_inputs_normalized = get_fgsm_input(inputs, self.epsilon)

            # Re-classify the perturbed image
            outputs = model(perturbed_inputs_normalized)

            loss_value += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                loss_value / (batch_idx + 1), 100. * correct / total, correct, total
            )
            progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return loss_value / len(data_loader)


# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]) -> Tensor:
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def get_fgsm_input(inputs: Tensor, epsilon: float):
    # Collect ``datagrad``
    data_grad = inputs.grad.data

    # Restore the data to its original scale
    data_denorm = denorm(inputs)

    # Call FGSM Attack
    perturbed_inputs = fgsm_attack(data_denorm, epsilon, data_grad)

    # Reapply normalization
    perturbed_inputs_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_inputs)

    return perturbed_inputs_normalized
