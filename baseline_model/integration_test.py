import unittest

from torch.nn import Module
from torch.utils.data import Dataset

from dataset import dataset
from model import model_selector
from scenario.base_scenario import BaseScenario
from scenario.fgsm_attack_scenario import FgsmAttackScenario
from scenario.pgd_attack_scenario import PgdAttackScenario
from scenario.scenario import Scenario


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.model: Module = model_selector.get_default_resnet()
        self.train_set: Dataset = dataset.get_random_cifar10_dataset(True, download=True)
        self.test_set: Dataset = dataset.get_default_cifar10_dataset(False, download=True)

    def test_base_scenario(self):
        """
        Default Res18 model should have a base accuracy more than 5%
        """
        scenario: Scenario = BaseScenario(model=self.model, train_set=self.train_set, test_set=self.test_set)
        result = scenario.perform(0)
        self.assertGreater(0.05, result['accuracy'])
        self.assertGreater(result['accuracy'], 0.2)

    def test_fgsm_attack_scenario(self):
        """
        Default Res18 model should have a base accuracy below 5% under FGSM attack
        """
        scenario: Scenario = FgsmAttackScenario(model=self.model, train_set=self.train_set, test_set=self.test_set)
        result = scenario.perform(0)
        self.assertGreater(0, result['accuracy'])
        self.assertGreater(result['accuracy'], 0.05)

    def test_pgd_attack_scenario(self):
        """
        Default Res18 model should have a base accuracy below 5% under PGD attack
        """
        scenario: Scenario = PgdAttackScenario(model=self.model, train_set=self.train_set, test_set=self.test_set)
        result = scenario.perform(0)

        self.assertGreaterEqual(0, result['accuracy'])
        self.assertGreater(result['accuracy'], 0.15)

    def test_comparison_fgsm_pgd(self):
        """
        FGSM and PGD should have same performance when PGD alpha=epsilon and noise_epochs = 1
        """
        fgsm_scenario: Scenario = FgsmAttackScenario(model=self.model, train_set=self.train_set, test_set=self.test_set,
                                                     epsilon=0.07)
        fgsm_result = fgsm_scenario.perform(0)

        pgd_scenario: Scenario = PgdAttackScenario(model=self.model, train_set=self.train_set, test_set=self.test_set,
                                                   epsilon=0.07,
                                                   alpha=0.07, noise_epochs=1)
        pgd_result = pgd_scenario.perform(0)

        self.assertEqual(fgsm_result['accuracy'], pgd_result['accuracy'])

    def test_comparison_FgsmPgdBase(self):
        """
        No attack should be allowed when epsilon = 0 or noise_epoch = 0
        """
        base_scenario: Scenario = BaseScenario(model=self.model, train_set=self.train_set, test_set=self.test_set)
        base_result = base_scenario.perform(0)

        fgsm_scenario: Scenario = FgsmAttackScenario(model=self.model, train_set=self.train_set, test_set=self.test_set,
                                                     epsilon=0)
        fgsm_result = fgsm_scenario.perform(0)

        pgd_scenario: Scenario = PgdAttackScenario(model=self.model, train_set=self.train_set, test_set=self.test_set,
                                                   noise_epochs=0)
        pgd_result = pgd_scenario.perform(0)

        self.assertEqual(fgsm_result['accuracy'], pgd_result['accuracy'])
        self.assertEqual(fgsm_result['accuracy'], base_result['accuracy'])


if __name__ == '__main__':
    unittest.main()
