import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from attack.fgsm import FGSM
from attack.identity import Identity
from attack.pgd import PGD
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
        train_dataset: Dataset = dataset.get_cifar10_dataset(True, True)
        dl = DataLoader(train_dataset)
        self.assertEqual(len(dl), 50000)

        test_dataset: Dataset = dataset.get_cifar10_dataset(False, True)
        dl = DataLoader(test_dataset)
        self.assertEqual(len(dl), 10000)

    def test_attack_FGSM_PGD_equal(self):
        """
        The FGSM and PGD should give equivalent output when PGD has carefully set the parameters
        """
        # setup
        train_dataset: Dataset = dataset.get_cifar10_dataset(True, True)
        dl = DataLoader(train_dataset, batch_size=4)
        inputs, labels = next(iter(dl))
        model: nn.Module = model_selector.get_default_resnet()

        perturbed_fgsm = FGSM(model, epsilon=0.007)(inputs, labels)
        perturbed_pgd = PGD(model, epsilon=0.007, alpha=0.007, noise_epochs=1)(inputs, labels)
        torch.all(torch.eq(perturbed_fgsm, perturbed_pgd)).item()

    def test_attack_FGSM_PGD_Identity_equal(self):
        """
        The FGSM and PGD should have no effect when the parameters are carefully set
        """
        # setup
        train_dataset: Dataset = dataset.get_cifar10_dataset(True, True)
        dl = DataLoader(train_dataset, batch_size=4)
        inputs, labels = next(iter(dl))
        model: nn.Module = model_selector.get_default_resnet()

        perturbed_fgsm = FGSM(model, epsilon=0)(inputs, labels)
        perturbed_pgd_1 = PGD(model, epsilon=0)(inputs, labels)
        perturbed_pgd_2 = PGD(model, noise_epochs=0)(inputs, labels)
        normal = Identity()(inputs, labels)
        torch.all(torch.eq(normal, perturbed_fgsm)).item()
        torch.all(torch.eq(normal, perturbed_pgd_1)).item()
        torch.all(torch.eq(normal, perturbed_pgd_2)).item()



if __name__ == '__main__':
    unittest.main()
