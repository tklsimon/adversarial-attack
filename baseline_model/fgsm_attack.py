from argparse import ArgumentParser

from torch.nn import Module
from torch.utils.data import Dataset

from dataset import dataset
from model import model_selector
from scenario.base_train_test_scenario import BaseTrainTestScenario
from scenario.fgsm_attack_scenario import FgsmAttackScenario

if __name__ == '__main__':
    parser = ArgumentParser(description='FGSM Attack Test')
    # model parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    # train and test parameters
    parser.add_argument('--dry_run', default=False, action='store_true', help='will not train or test model')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')
    # model parameter
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    args = parser.parse_args()

    print("*** train-test-load script ***")

    # initialize scenario
    model: Module = model_selector.get_default_resnet(layers=args.layers)
    train_set: Dataset = dataset.get_random_cifar10_dataset(True, download=args.load_data)
    test_set: Dataset = dataset.get_default_cifar10_dataset(False, download=args.load_data)
    scenario: BaseTrainTestScenario = FgsmAttackScenario(load_path=args.load_path,
                                                        batch_size=args.batch_size,
                                                        model=model,
                                                        train_set=train_set,
                                                        test_set=test_set)
    print("*** arguments: ***")
    print(args)
    print("*** scenario: ***")
    print(scenario)
    print()

    if not args.dry_run:
        scenario.train_eval_test_save(0)
