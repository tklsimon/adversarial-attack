import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from .train_test_scenario import TrainTestScenario


class PgdAttackScenario(TrainTestScenario):
    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, train_eval_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.03,
                 alpha: float = 0.007, num_iter: int = 10):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, train_eval_ratio=train_eval_ratio,
                         model=model, train_set=train_set, test_set=test_set)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: _Loss) -> float:
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

            # get PGD noise inputs
            perturbed_inputs_normalized = pgd_attack(inputs,  self.epsilon, self.alpha, self.num_iter)

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


def pgd_attack(inputs: Tensor, epsilon: float, alpha:float , num_iter:int) -> Tensor:
    #Set up loss for iteration maximising loss
    criterion = nn.CrossEntropyLoss()
    #Copying tensor from original for operation
    perturbing_input = inputs.clone().detach()

    for i in range(self.num_iter): #default epsilon: float = 0.03, alpha: float = 0.007, num_iter: int = 10
        #enable grad computation
        perturbing_input.requires_grad = True
        #makes prediction on perturbing images
        outputs = self.model(perturbing_input)
        #clear model grad before computing
        self.model.zero_grad()
        #Calculate loss
        loss = criterion(outputs, targets)
        #Compute gradient
        loss.backward()
        #learning rate(alpha)*neg(grad) to maximise loss
        adv_images = perturbing_input + self.alpha * perturbing_input.grad.sign()
        #confining perturbation
        eta = torch.clamp(adv_images - perturbing_input.data, min=-self.epsilon, max=self.epsilon)
        #output perturbed image for next iteration
        perturbing_input = torch.clamp(perturbing_input.data + eta, min=0, max=1).detach()
    return perturbing_input
