import os
from abc import ABC

import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from tqdm import tqdm

from .base_train_test_scenario import BaseTrainTestScenario


class TrainTestScenario(BaseTrainTestScenario, ABC):

    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, train_eval_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, train_eval_ratio=train_eval_ratio,
                         model=model, train_set=train_set, test_set=test_set)

        # initialize objects
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_data()
        self._init_model()

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

    def train(self, epoch: int = 1):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=len(self.train_loader),
                                                        epochs=epoch)
        criterion = torch.nn.CrossEntropyLoss()

        for i in range(epoch):
            print('==> Train Epoch: %d..' % i)

            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device_name), targets.to(self.device_name)

                optimizer.zero_grad()
                outputs = self.model(inputs)
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

            scheduler.step()

    def test(self, model: Module, device_name: str, data_loader: DataLoader, criterion: _Loss) -> float:
        model.eval()  # switch to evaluation mode
        loss_value = 0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(device_name), targets.to(device_name)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss_value += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    loss_value / (batch_idx + 1), 100. * correct / total, correct, total
                )
                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
        return loss_value / len(data_loader)

    def save(self, state_dict: dict, save_path: str, train_param: str):
        print('==> Save to checkpoint..', save_path)
        augmented_path = os.path.join("./checkpoint", save_path)
        checkpoint_dir: str = os.path.dirname(augmented_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        data = {'state_dict': state_dict, 'param_dict': train_param}
        torch.save(data, augmented_path)

    def train_eval_test_save(self, epoch: int = 1):
        save_best = True if self.save_path else False
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=len(self.train_loader),
                                                        epochs=epoch)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_score = 0
        best_model_state_dict: dict = dict()
        for i in range(epoch):
            print('==> Train Epoch: %d..' % i)

            """train"""
            self.model.train()  # switch to train mode
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device_name), targets.to(self.device_name)

                optimizer.zero_grad()
                outputs = self.model(inputs)
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
                break

            """evaluation"""
            if self.validation_loader is not None and len(self.validation_loader) > 0:
                eval_loss: float = self.test(self.model, self.device_name, self.validation_loader, criterion)
                # scheduler.step(eval_loss))
            scheduler.step()

            if save_best:
                if 100. * correct / total > best_val_score:
                    best_val_score = 100. * correct / total
                    best_model_state_dict = self.model.state_dict()

        """test"""
        print('==> Test')
        test_loss = self.test(self.model, self.device_name, self.test_loader, criterion)

        """save"""
        if save_best:
            self.save(best_model_state_dict, self.save_path, str(self))

        torch.cuda.empty_cache()
