import os
from random import random

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader, Subset
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

    def __str__(self):
        return "model=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, weigh_decay=%.2E, momentum=%.2E, " \
               "train_eval_ratio=%.2E" % (
                   self.model.__class__.__name__,
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.train_eval_ratio
               )

    def _init_data(self):
        print('==> Preparing data..')

        """split into train-eval set"""
        # Calculate the number of samples for each split
        num_samples = len(self.train_set)
        train_size = int(self.train_eval_ratio * num_samples)

        # Create indices for train and validation sets
        indices = list(range(num_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create subsets for train and validation sets
        train_dataset = Subset(self.train_set, train_indices)
        val_dataset = Subset(self.train_set, val_indices)

        # split into validation and train set
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        if self.train_eval_ratio < 1:
            self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        else:
            self.validation_loader = None
        print("target class available: ", self.classes)
        print("no. of train batch: ", len(train_indices))
        print("no. of validation batch: ", len(val_indices))
        print("no. of test batch: ", len(self.test_loader))

    def _init_model(self):
        print('==> Building model..')
        self.model = self.model.to(self.device_name)
        if self.device_name == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        if self.load_path:
            print('==> Resuming from checkpoint ', self.load_path)
            augmented_path = os.path.join("./checkpoint", self.load_path)
            checkpoint_dir: str = os.path.dirname(augmented_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint = torch.load(augmented_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            if 'param_dict' in checkpoint:
                print("==> Loaded model: ", checkpoint['param_dict'])

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
                    perturbed_inputs_normalized = pgd_attack(inputs, self.epsilon, self.alpha, self.num_iter)
                    outputs = model(perturbed_inputs_normalized)
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
