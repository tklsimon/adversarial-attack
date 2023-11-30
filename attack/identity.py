import torch


class Identity(torch.nn.Module):
    """
    No processing for input image
    """

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return inputs.detach()

    def __str__(self):
        return "Attack=%s" % self.__name__
