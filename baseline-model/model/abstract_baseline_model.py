import os
from abc import ABC

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

from .baseline_model import BaselineModel


class AbstractBaselineModel(BaselineModel, ABC):

    def __init__(self, model, resume: bool = False, lr: float = 0.001, batch_size: int = 4, momentum: float = 0.9,
                 weight_decay: float = 0):
        super().__init__(model, resume, lr, batch_size, momentum, weight_decay)

        # initialize objects
        self._init_data()
        self._init_model()

    def __str__(self):
        return "model=%s, resume=%s, batch_size=%d, lr=%.3f, weigh_decay=%.3f, momentum=%.3f" % (
            self.model.__class__.__name__, self.resume, self.batch_size, self.lr, self.weight_decay, self.momentum
        )

    def _init_data(self):
        print('==> Preparing data..')

        train_set = self._set_train_set()
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = self._set_test_set()
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def _init_model(self):
        print('==> Building model..')
        # make it able to override
        self.model = self._set_model()
        self.model.fc = nn.Linear(512, 10)
        self.model = self.model.to(self.device_name)
        if self.device_name == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        if self.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpoint['model'])

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = torch.nn.CrossEntropyLoss()
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

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device_name), targets.to(self.device_name)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total
                )
                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))
