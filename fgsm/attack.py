import torch
from torch import Tensor


# def fgsm_attack_batch(data: Tensor, epsilon: float, batch_size: int):
#     for t in data:
#         a = batch_size(t, epsilon)
#     return perturbed_image


def fgsm_attack(data: Tensor, epsilon: float) -> Tensor:
    # Restore the data to its original scale
    image = denorm(data)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081], device: str = 'cuda') -> Tensor:
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
        :param device: cuda or cpu
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
