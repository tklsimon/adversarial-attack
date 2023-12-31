import copy
import os
from abc import ABC
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm

from .scenario import Scenario


class BaseScenario(Scenario, ABC):
    """
    Implementation of Base Scenario
    This class contains basic training function for baseline model
    """

    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 64,
                 momentum: float = 0.9, weight_decay: float = 0, test_val_ratio: float = 0.5, soft_label: float = 0.0,
                 model: nn.Module = None, attacker: nn.Module = None, train_set: Dataset = None,
                 test_set: Dataset = None):
        """
        Constructor of BaseScenario

        :param load_path: model weight's path under checkpoint folder
        :param save_path: path to save trained model's weight under checkpoint folder
        :param lr: learning rate
        :param batch_size: batch size of processing data, use in train and test
        :param momentum: optimizer settings
        :param weight_decay: optimizer settings
        :param test_val_ratio: ratio of test dataset : validation dataset.  If set to 1, then all data are for testing
        :param soft_label: label smoothing factor
        :param model: model to be trained / tested
        :param train_set: train dataset
        :param test_set: test dataset
        """
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, test_val_ratio=test_val_ratio, soft_label=soft_label,
                         model=model, train_set=train_set, test_set=test_set)

        # initialize objects
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.previous_params: List = []
        self.attacker: nn.Module = attacker
        self._init_data()
        self._init_model()

    def __str__(self):
        return "Scenario=%s, model=%s, attacker=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, " \
               "weigh_decay=%.2E, momentum=%.2E, test_val_ratio=%.2E, soft_label=%.2E" % (
                   self.__class__.__name__,
                   self.model.__class__.__name__,
                   str(self.attacker),
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.test_val_ratio, self.soft_label)

    def _init_data(self):
        # Initialize data, including test and validation split, and loader
        print('==> Preparing data..')

        # Split into test and val set
        # Number of sample for test set
        num_samples = len(self.test_set)
        test_size = int(self.test_val_ratio * num_samples)

        # Indices of test and validation sets
        indices = list(range(num_samples))
        test_indices = indices[:test_size]
        val_indices = indices[test_size:]

        # Split into test and validation sets
        test_dataset = Subset(self.test_set, test_indices)
        val_dataset = Subset(self.test_set, val_indices)

        # Train loader, test loader and validation loader
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        if self.test_val_ratio < 1:
            self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        else:
            self.validation_loader = None

        print("no. of train batch: ", len(self.train_loader))
        print("no. of validation batch: ", len(self.validation_loader))
        print("no. of test batch: ", len(self.test_loader))

    def _init_model(self):
        # initialize the model, such as loading weights from checkpoints
        print('==> Initializing model..')
        self.model = self.model.to(self.device_name)
        self.model = torch.nn.DataParallel(self.model)
        if self.device_name == 'cuda':
            torch.backends.cudnn.benchmark = True
        # Load checkpoint model if load_path is not NULL
        if self.load_path:
            print('==> Resuming from checkpoint ', self.load_path)
            augmented_path = self._get_or_create_checkpoint_path(self.load_path)
            checkpoint = torch.load(augmented_path, map_location=self.device_name)
            self.model.load_state_dict(checkpoint['state_dict'])
            if 'param_dict' in checkpoint:
                print("==> Loaded model: ")
                self.previous_params = eval(checkpoint['param_dict'])
                for i in range(len(self.previous_params)):
                    print("==> [%2d] %s" % (i, self.previous_params[i]))

    def train(self, model: nn.Module, device_name: str, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer, scheduler, criterion, save_best: bool = False, epoch: int = 1):
        best_val_score = 0
        best_model_state_dict: dict = dict()
        best_epoch: int = 0
        for i in range(epoch):
            print('==> Training Epoch: %d..' % i)

            # Training Part
            model.train()  # Switch to train mode
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                inputs = inputs.to(device_name)
                targets = targets.to(device_name)

                perturbed_inputs = self.attack(inputs, targets)

                optimizer.zero_grad()
                outputs = model(perturbed_inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total)

                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))

            # Validation Part
            val_loss: Dict = {}
            if self.validation_loader is not None and len(validation_loader) > 0:
                val_loss = self.test(model, device_name, validation_loader, criterion)
            scheduler.step()

            if save_best:
                if 'accuracy' in val_loss.keys():
                    if val_loss['accuracy'] > best_val_score:
                        print("==> current best epoch = %d" % i)
                        best_val_score = val_loss['accuracy']
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = i
                else:
                    best_epoch = i

        # Save trained model to checkpoint
        if save_best:
            self.previous_params.append(str(self) + ", best epoch=" + str(best_epoch))
            self.save(best_model_state_dict, self.save_path, self.previous_params)

    def test(self, model: nn.Module, device_name: str, data_loader: DataLoader, criterion: nn.Module) -> Dict:
        model.eval()  # switch to evaluation mode
        loss_value = 0
        correct = 0
        total = 0
        similarities = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress_bar:
                inputs = inputs.to(device_name)
                targets = targets.to(device_name)
                perturbed_input = self.attack(inputs, targets)
                outputs = model(perturbed_input)
                loss = criterion(outputs, targets)

                loss_value += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                similarities += similarity(inputs, perturbed_input)

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    loss_value / (batch_idx + 1), 100. * correct / total, correct, total
                )
                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return {'average test_loss': loss_value / len(data_loader), 'accuracy': correct / total,
                'average similarity': similarities / total}

    def attack(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        transform input to noise image

        :param inputs: original input
        :param targets: original target
        :return: input with noise
        """
        if self.attacker is None:
            return inputs.detach()
        return self.attacker(inputs, targets)

    def save(self, state_dict: dict, save_path: str, train_params: List):
        """Save model

        :param train_params: all the previous and current train parameters
        :param state_dict: the weightings of the model
        :param save_path: where to save the model
        """
        print('==> Save to checkpoint..', save_path)
        augmented_path = self._get_or_create_checkpoint_path(save_path)
        data = {'state_dict': state_dict, 'param_dict': str(train_params)}
        torch.save(data, augmented_path)

    def perform(self, epoch: int = 1) -> Dict:
        save_best = True if self.save_path else False

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=len(self.train_loader),
                                                        epochs=epoch)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.soft_label)

        # train, validation and save best model
        self.train(self.model, self.device_name, self.train_loader, self.validation_loader, optimizer, scheduler,
                   criterion, save_best, epoch)

        # Test Part
        print('==> Testing')
        test_metric = self.test(self.model, self.device_name, self.test_loader, criterion)

        torch.cuda.empty_cache()
        return test_metric

    def _get_or_create_checkpoint_path(self, input_path) -> str:
        augmented_path = os.path.join("checkpoint", input_path)
        checkpoint_dir: str = os.path.dirname(augmented_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return augmented_path


def similarity(original_images: torch.Tensor, adversarial_images: torch.Tensor):
    # Move images back to CPU for visualization
    original_images = original_images.cpu().numpy()
    adversarial_images = adversarial_images.detach().cpu().numpy()

    # Compute cosine similarity
    images_flatten = original_images.reshape(original_images.shape[0], -1)
    adversarial_images_flatten = adversarial_images.reshape(adversarial_images.shape[0], -1)

    cosine_similarities = torch.nn.functional.cosine_similarity(torch.from_numpy(images_flatten),
                                                                torch.from_numpy(adversarial_images_flatten))
    return cosine_similarities.sum().item()
