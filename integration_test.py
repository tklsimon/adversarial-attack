import unittest

import torch.nn as nn
from torch.utils.data import Dataset

from dataset import dataset
from model import model_selector
from scenario.scenario import Scenario
from scenario.base_scenario import BaseScenario


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.model: nn.Module = model_selector.get_default_resnet()
        self.train_set: Dataset = dataset.get_cifar10_dataset(True, download=True)
        self.test_set: Dataset = dataset.get_cifar10_dataset(False, download=True)

    def test_base_scenario(self):
        """
        Default Res18 model should have a base accuracy more than 5%
        """
        scenario: Scenario = BaseScenario(model=self.model, train_set=self.train_set, test_set=self.test_set)
        result = scenario.perform(0)
        print("Test Accuracy: ", result['accuracy'])
        self.assertGreater(result['accuracy'], 0.05)
        self.assertGreater(0.2, result['accuracy'])


if __name__ == '__main__':
    unittest.main()
