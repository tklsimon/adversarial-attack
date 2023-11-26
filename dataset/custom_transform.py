import torch


class MaskTransform(torch.nn.Module):
    """
    Mask input images
    """
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_prob = mask_prob

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(tensor.size()) < self.mask_prob
        return tensor * mask.float()
