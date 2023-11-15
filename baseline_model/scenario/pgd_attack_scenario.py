import os
from random import random

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from .base_scenario import BaseScenario


class PgdAttackScenario(BaseScenario):
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

    def __str__(self):
        return "model=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, weigh_decay=%.2E, momentum=%.2E, " \
               "train_eval_ratio=%.2E, epsilon=%.2E, alpha=%.2E, num_iter=%d" % (
                   self.model.__class__.__name__,
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.train_eval_ratio, self. epsilon, self.alpha, self.num_iter
               )

    def train(self, model: Module, device_name: str, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer, scheduler, criterion, save_best: bool = False, epoch: int = 1):
        best_val_score = 0
        best_model_state_dict: dict = dict()
        for i in range(epoch):
            print('==> Train Epoch: %d..' % i)

            """train"""
            model.train()  # switch to train mode
            train_loss = 0
            correct = 0
            total = 0

            """evaluation"""
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(device_name), targets.to(device_name)

                optimizer.zero_grad()

                rand_num = random.random()

                if rand_num < 0.5:
                    # 50% chance to perform PGD attack
                    perturbed_inputs = pgd_attack(inputs, self.epsilon, self.alpha, self.num_iter)
                    outputs = model(perturbed_inputs)
                else:
                    # 50% chance to just classify the original inputs
                    outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total
                )

                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))

            """evaluation"""
            if self.validation_loader is not None and len(validation_loader) > 0:
                eval_loss: float = self.test(model, device_name, validation_loader, criterion)
                # scheduler.step(eval_loss))
            scheduler.step()

            if save_best:
                if 100. * correct / total > best_val_score:
                    best_val_score = 100. * correct / total
                    best_model_state_dict = model.state_dict()

        """save"""
        if save_best:
            self.save(best_model_state_dict, self.save_path, str(self))

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
            perturbed_inputs = pgd_attack(inputs, targets, self.model, self.epsilon, self.alpha,
                                          self.num_iter)

            # Re-classify the perturbed image
            outputs = model(perturbed_inputs)

            loss_value += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                loss_value / (batch_idx + 1), 100. * correct / total, correct, total
            )
            progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return loss_value / len(data_loader)


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
