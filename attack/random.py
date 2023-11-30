import torch


class RandomNoiseAttack(torch.nn.Module):
    """
    No processing for input image
    """

    def __init__(self, epsilon: float = 0.007):
        super().__init__()
        self.epsilon = epsilon

    def __str__(self):
        return "Attack=%s (epsilon=%.5f)" % (self.__name__, self.epsilon)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon: float = self.epsilon
        variance: int = 3
        noise = torch.randn(*inputs.shape).to(self.device_name) * variance
        noise = torch.clamp(noise, -1, 1)
        noise = noise * epsilon
        return inputs + noise
