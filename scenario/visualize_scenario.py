import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from typing import Dict

from .base_scenario import BaseScenario


class visualizeScenario(BaseScenario):

    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, test_val_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.03,
                 alpha: float = 0.007, num_iter: int = 10, attack_mode: str = None):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, test_val_ratio=test_val_ratio,
                         model=model, train_set=train_set, test_set=test_set)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.attack_mode = attack_mode

    def __str__(self):
        return "model=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, weigh_decay=%.2E, momentum=%.2E, " \
               "test_val_ratio=%.2E, epsilon=%.2E, alpha=%.2E, num_iter=%d" % (
                   self.model.__class__.__name__,
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.test_val_ratio, self.epsilon, self.alpha, self.num_iter
               )

    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: nn.Module) -> Dict:
        model.eval()  # switch to evaluation mode
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device_name), labels.to(device_name)

        # Perform the fgsm/PGD attack
        if self.attack_mode == 'fgsm':
            adversarial_images = fgsm_attack(images, self.epsilon, labels, model)
        else:
            adversarial_images = pgd_attack(images, labels, self.model, self.epsilon, self.alpha,
                                            self.num_iter)

        # Move images back to CPU for visualization
        images = images.cpu().numpy()
        adversarial_images = adversarial_images.detach().cpu().numpy()

        # Un-normalize the images for visualization
        original_images = (images + 1) / 2.0
        adversarial_images = (adversarial_images + 1) / 2.0

        # Compute cosine similarity
        images_flatten = original_images.reshape(original_images.shape[0], -1)
        adversarial_images_flatten = adversarial_images.reshape(adversarial_images.shape[0], -1)

        cosine_similarity = F.cosine_similarity(torch.from_numpy(images_flatten),
                                                torch.from_numpy(adversarial_images_flatten))
        cosine_similarity_mean = cosine_similarity.mean().item()
        print('Average Cosine Similarity:', cosine_similarity_mean)

        return {'average similarity': cosine_similarity_mean}

    '''
    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: _Loss) -> float:
        model.eval()  # switch to evaluation mode
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device_name), labels.to(device_name)

        # Perform the PGD attack
        adversarial_images = pgd_attack(images, labels, self.model, self.epsilon, self.alpha,
                                          self.num_iter)


        # Move images back to CPU for visualization
        images = images.cpu().numpy()
        adversarial_images = adversarial_images.detach().cpu().numpy()

        # Un-normalize the images for visualization
        images = (images + 1) / 2.0
        adversarial_images = (adversarial_images + 1) / 2.0
        
        print(images.shape)  # Should be (batch_size, 3, height, width)
        print(np.min(images), np.max(images))  # Should be between 0 and 1
        
        fig, axs = plt.subplots(2, 10, figsize=(25, 5))
        
        for i in range(10):  # Display 10 images
            # Display original images
            axs[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
            axs[0, i].axis('off')
            

            # Display adversarial images
            axs[1, i].imshow(np.transpose(adversarial_images[i], (1, 2, 0)))
            axs[1, i].axis('off')
        
        axs[0, 0].set_ylabel("Original", size='large')
        axs[1, 0].set_ylabel("Adversarial", size='large')
        
        
        
        plt.show()  
      '''


def pgd_attack(inputs: Tensor, targets, model: Module, epsilon: float, alpha: float, num_iter: int) -> Tensor:
    # Set up loss for iteration maximising loss
    criterion = nn.CrossEntropyLoss()
    # Copying tensor from original for operation
    perturbing_input = inputs.clone().detach()

    for i in range(num_iter):  # default epsilon: float = 0.03, alpha: float = 0.007, num_iter: int = 10
        # enable grad computation
        perturbing_input.requires_grad = True
        # makes prediction on perturbing images
        outputs = model(perturbing_input)
        # clear model grad before computing
        model.zero_grad()
        # Calculate loss
        loss = criterion(outputs, targets)
        # Compute gradient
        loss.backward()
        # learning rate(alpha)*neg(grad) to maximise loss
        adv_images = perturbing_input + alpha * perturbing_input.grad.sign()
        # confining perturbation
        eta = torch.clamp(adv_images - perturbing_input.data, min=-epsilon, max=epsilon)
        # output perturbed image for next iteration
        perturbing_input = torch.clamp(perturbing_input.data + eta, min=0, max=1).detach()
    return perturbing_input


def fgsm_attack(inputs: Tensor, epsilon: float, labels, model: Module) -> Tensor:
    # Collect ``datagrad``
    perturbing_input = inputs.clone().detach()
    perturbing_input.requires_grad = True

    # Perform a forward pass
    outputs = model(perturbing_input)

    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    # Use loss to perform a backward pass
    model.zero_grad()
    loss.backward()

    # Now `perturbing_input.grad` will not be None
    data_grad = perturbing_input.grad.data

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_input = inputs + epsilon * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_input = torch.clamp(perturbed_input, 0, 1)

    # Return the perturbed image
    return perturbed_input
