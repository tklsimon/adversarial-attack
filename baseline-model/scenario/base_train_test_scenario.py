"""
Import cifar10 image data with PyTorch
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
from abc import abstractmethod, ABC

from torch.nn import Module
from torch.utils.data import Dataset


class BaseTrainTestScenario(ABC):

    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, train_eval_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None):
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

        # dataset dependent
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # train and test parameter
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = None
        self.test_loader = None
        self.validation_loader = None

    """override this method for train model"""

    @abstractmethod
    def train_eval_test_save(self, epoch: int = 1, save_best: bool = False):
        pass

