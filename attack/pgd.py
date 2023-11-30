import torch


class PGD(torch.nn.Module):
    """
    PGD
    """

    def __init__(self, module: torch.nn.Module, epsilon: float = 0.007, alpha: float = 0.03, noise_epochs: int = 10):
        super().__init__()
        self.module = module
        self.epsilon = epsilon
        self.alpha = alpha
        self.noise_epochs = noise_epochs

    def __str__(self):
        return "Attack=%s (epsilon=%.5f, alpha=%.5f, noise_epochs=%.5f)" % \
               (self.__name__, self.epsilon, self.alpha, self.noise_epochs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            # initialize attack settings
            _inputs = inputs.clone().detach()
            _targets = targets.clone().detach()
            criterion = torch.nn.CrossEntropyLoss()

            perturbed_input = inputs.clone().detach()
            for i in range(self.noise_epochs):
                # enable grad for inputs
                perturbed_input.requires_grad = True

                # Forward pass the data through the model
                output = self.model(perturbed_input)

                # Calculate the loss
                loss: torch.Tensor = criterion(output, _targets)

                # Zero all existing gradients
                self.model.zero_grad()

                # Calculate gradients of model in backward pass
                loss.backward()

                # Create the perturbed image by adjusting each pixel of the input image
                perturbed_input = perturbed_input.detach() + self.alpha * perturbed_input.grad.sign()
                # confining perturbation
                eta = torch.clamp(perturbed_input - _inputs, - self.epsilon, self.epsilon)
                # Adding clipping to maintain [0,1] range
                perturbed_input = torch.clamp(_inputs + eta, 0, 1)

            # Return the perturbed image
            return perturbed_input.detach()
