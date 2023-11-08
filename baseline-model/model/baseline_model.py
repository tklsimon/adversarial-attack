"""
Import cifar10 image data with PyTorch
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
from abc import abstractmethod, ABC

import torch
from torch.nn import Module
from torch.utils.data import Dataset


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

    """override this method for PyTorch model"""
    @abstractmethod
    def _set_model(self) -> Module:
        pass

    """override this method for train dataset"""
    @abstractmethod
    def _set_train_set(self) -> Dataset:
        pass

    """override this method for test dataset"""
    @abstractmethod
    def _set_test_set(self) -> Dataset:
        pass

    """override this method for train model"""
    @abstractmethod
    def train(self):
        pass

    """override this method for test model"""
    @abstractmethod
    def test(self):
        pass
