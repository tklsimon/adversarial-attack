import torch


class RandomNoiseAttack(torch.nn.Module):
    """
    No processing for input image
    """

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon: float = self.epsilon
        variance: int = 3
        noise = torch.randn(*inputs.shape).to(self.device_name) * variance
        noise = torch.clamp(noise, -1, 1)
        noise = noise * epsilon
        return inputs + noise



