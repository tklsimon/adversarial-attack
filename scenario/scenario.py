"""
Import cifar10 image data with PyTorch
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
from abc import abstractmethod, ABC
from typing import Dict

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Scenario(ABC):
    """Interface of the Scenario Class.  A default scenario should contain the below stages:
    load model, repeat (train, validation), save model and test"""

    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, test_val_ratio: float = 0.5,
                 model: nn.Module = None, train_set: Dataset = None, test_set: Dataset = None):
        """Constructor of Scenario

        :param load_path: model weight's path under checkpoint folder
        :param save_path: path to save trained model's weight under checkpoint folder
        :param lr: learning rate
        :param batch_size: batch size of processing data, use in train and test
        :param momentum: optimizer settings
        :param weight_decay: optimizer settings
        :param test_val_ratio: ratio of train dataset : validation dataset.  If set to 1, then train with all data
        :param model: model to be trained / tested
        :param train_set: train dataset
        :param test_set: test dataset
        """
        # set model
        self.model: nn.Module = model

        # model parameter
        self.device_name: str = ''
        self.load_path: str = load_path
        self.save_path: str = save_path
        self.batch_size: int = batch_size
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.test_val_ratio: float = test_val_ratio

        # train and test parameter
        self.train_set: Dataset = train_set
        self.test_set: Dataset = test_set
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.validation_loader: DataLoader = None

    @abstractmethod
    def perform(self, epoch: int = 1) -> Dict:
        """
        Perform the scenario

        :param epoch: number of training iteration
        :return Dict: test metrics
        """
        pass
