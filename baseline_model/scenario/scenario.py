"""
Import cifar10 image data with PyTorch
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
from abc import abstractmethod, ABC

from torch.nn import Module
from torch.utils.data import Dataset


class Scenario(ABC):
    """Interface of the Scenario Class.  A default scenario should contains the below stages:
    load model, repeat (train, evaluate), save model and test"""
    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, train_eval_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None):
        """Constructor of Scenario

        :param load_path: model weight's path under checkpoint folder
        :param save_path: path to save trained model's weight under checkpoint folder
        :param lr: learning rate
        :param batch_size: batch size of processing data, use in train and test
        :param momentum: optimizer settings
        :param weight_decay: optimizer settings
        :param train_eval_ratio: ratio of train dataset : evaluation dataset.  If set to 1, then all data are for training
        :param model: model to be trained / tested
        :param train_set: train dataset
        :param test_set: test dataset
        """
        # set model
        self.model = model

        # model parameter
        self.device_name = ''
        self.load_path = load_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_eval_ratio = train_eval_ratio

        # train and test parameter
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = None
        self.test_loader = None
        self.validation_loader = None

    @abstractmethod
    def perform(self, epoch: int = 1):
        """
        Perform the scenario

        :param epoch: number of training iteration
        """
        pass

