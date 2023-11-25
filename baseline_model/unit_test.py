import unittest
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import dataset
from model import model_selector


class TestUnitTest(unittest.TestCase):
    """
    Perform Unit Test
    """

    def test_model(self):
        for layer in [18, 34, 50, 101, 152]:
            model: nn.Module = model_selector.get_default_resnet(layers=layer)
            self.assertIsNotNone(model)

    def test_dataset(self):
        train_dataset: Dataset = dataset.get_default_cifar10_dataset(True, True)
        dl = DataLoader(train_dataset)
        self.assertEqual(len(dl), 50000)

        test_dataset: Dataset = dataset.get_default_cifar10_dataset(False, True)
        dl = DataLoader(test_dataset)
        self.assertEqual(len(dl), 10000)


if __name__ == '__main__':
    unittest.main()
