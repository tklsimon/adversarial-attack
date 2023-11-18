import os
import sys
from argparse import ArgumentParser
from typing import Dict

from torch.nn import Module
from torch.utils.data import Dataset

from dataset import dataset
from model import model_selector

par_dir: str = os.path.dirname(os.getcwd())
sys.path.append(par_dir)

from scenario.scenario import Scenario  # noqa
from scenario.pgd_defense_scenario import PgdDefenseScenario  # noqa

if __name__ == '__main__':
    parser = ArgumentParser(description='PGD Grid Search Script')
    # model parameters
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    # train and test parameters
    parser.add_argument('--train_epochs', default=10, type=int, help='no. of epochs for train')
    parser.add_argument('--train_val_ratio', default=0.99, type=float, help='ratio for train-eval split')
    parser.add_argument('--test_only', default=False, action='store_true', help='only test model')
    parser.add_argument('--dry_run', default=False, action='store_true', help='will not train or test model')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')
    # model parameter
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--save_path', default=None, type=str, help='save checkpoint')
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    parser.add_argument('--clean_model', default=True, action='store_false', help='load online pretrained parameters')
    parser.add_argument('--model_type', default='', type=str, help='custom or default model')
    # attack parameter
    parser.add_argument('--epsilon_start', default=0.00, type=float, help='PGD noise attack epsilon start')
    parser.add_argument('--epsilon_end', default=0.10, type=float, help='PGD noise attack epsilon end')
    parser.add_argument('--epsilon_steps', default=10, type=int, help='no. of PGD epsilon steps')
    parser.add_argument('--alpha', default=0.007, type=float, help='PGD noise attack alpha')
    parser.add_argument('--noise_epochs', default=10, type=int, help='no of epochs for PGD noise attack')
    args = parser.parse_args()

    print("*** pgd grid search script ***")
    print("*** arguments: ***")
    print(args)
    print()

    # initialize scenario
    if args.model_type == 'custom':
        model: Module = model_selector.get_custom_resnet(layers=args.layers)
    else:
        model: Module = model_selector.get_default_resnet(layers=args.layers, pretrain=args.clean_model)
    train_set: Dataset = dataset.get_random_cifar10_dataset(True, download=args.load_data)
    test_set: Dataset = dataset.get_default_cifar10_dataset(False, download=args.load_data)

    epsilon_stair: float = args.epsilon_end - args.epsilon_start
    grid_search_result = []
    for step in range(args.epsilon_steps):
        epsilon: float = args.epsilon_start + step * epsilon_stair
        print("*** iteration %d, epsilon = %.5f ***" % (step, epsilon))
        scenario: Scenario = PgdDefenseScenario(load_path=args.load_path,
                                                save_path=args.save_path,
                                                batch_size=args.batch_size,
                                                lr=args.lr,
                                                momentum=args.momentum,
                                                weight_decay=args.weight_decay,
                                                train_val_ratio=args.train_val_ratio,
                                                epsilon=epsilon,
                                                alpha=args.alpha,
                                                noise_epochs=args.noise_epochs,
                                                model=model,
                                                train_set=train_set,
                                                test_set=test_set)

        test_metric: Dict = 0
        if not args.dry_run:
            if args.test_only:
                test_metric = scenario.perform(0)
            else:
                test_metric = scenario.perform(args.train_epochs)
        grid_search_result.append((epsilon, test_metric))

    print("*** end result ***")
    for row in grid_search_result:
        print(row)
