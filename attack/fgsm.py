import torch


class FGSM(torch.nn.Module):
    """
    FGSM
    """

    def __init__(self, module: torch.nn.Module, epsilon: float = 0.007):
        super(FGSM, self).__init__()
        self.module = module
        self.epsilon = epsilon

    def __str__(self):
        return "Attack=%s (epsilon=%.2f)" % (self.__class__.__name__, self.epsilon)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            # initialize attack settings
            _inputs = inputs.clone().detach()
            _targets = targets.clone().detach()
            criterion = torch.nn.CrossEntropyLoss()

            # enable grad for inputs
            _inputs.requires_grad = True

            # Forward pass the data through the model
            output = self.module(_inputs)

            # Calculate the loss
            loss: torch.Tensor = criterion(output, _targets)

            # Zero all existing gradients
            self.module.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_input = _inputs.detach() + self.epsilon * _inputs.grad.sign()
            # Adding clipping to maintain [0,1] range
            perturbed_input = torch.clamp(perturbed_input, 0, 1)
            # Return the perturbed image
            return perturbed_input
