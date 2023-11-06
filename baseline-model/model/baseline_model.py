"""
Import cifar10 image data with PyTorch
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import os
from abc import abstractmethod, ABC

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet


class BaselineModel(ABC):

    def __init__(self, model, resume: bool = False, lr: float = 0.001, batch_size: int = 4, momentum: float = 0.9,
                 weight_decay: float = 0):
        self.model = model

        # model parameter
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume = resume
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # data dependent
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # train and test parameter
        self.train_loader = None
        self.test_loader = None

        # initialize objects
        self._init_data()
        self._init_model()

    def __str__(self):
        return "model=baseline model %s, resume=%s, batch_size=%d, lr=%.3f, weigh_decay=%.3f, momentum=%.3f" % (
            self.model.__class__.__name__, self.resume, self.batch_size, self.lr, self.weight_decay, self.momentum
        )

    def _init_data(self):
        print('==> Preparing data..')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def _init_model(self):
        print('==> Building model..')
        # make it able to override
        self.model = self.get_model()
        self.model = self.model.to(self.device_name)
        if self.device_name == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        if self.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpoint['model'])

    """override this method for the model type"""

    @abstractmethod
    def get_model(self) -> ResNet:
        pass

    def train(self, epoch: int):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = torch.nn.CrossEntropyLoss()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
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
            print('[batch %2d]     %s' % (batch_idx, log_msg))
