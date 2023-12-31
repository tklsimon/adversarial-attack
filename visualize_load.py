import os
import sys
from argparse import ArgumentParser

from torch.nn import Module
from torch.utils.data import Dataset

from dataset import dataset
from model import model_selector

par_dir: str = os.path.dirname(os.getcwd())
sys.path.append(par_dir)

from scenario.scenario import Scenario  # noqa
from scenario.visualize_scenario import visualizeScenario  # noqa

if __name__ == '__main__':
    parser = ArgumentParser(description='Visualization Load Script')
    # model parameters
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    # train and test parameters
    parser.add_argument('--train_epochs', default=10, type=int, help='no. of epochs for train')
    parser.add_argument('--test_val_ratio', default=0.99, type=float, help='ratio for train-eval split')
    parser.add_argument('--test_only', default=False, action='store_true', help='only test model')
    parser.add_argument('--dry_run', default=False, action='store_true', help='will not train or test model')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')
    # model parameter
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--save_path', default=None, type=str, help='save checkpoint')
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    parser.add_argument('--clean_model', default=True, action='store_false', help='load online pretrained parameters')
    # attack parameter
    parser.add_argument('--epsilon', default=0.03, type=float, help='PGD noise attack epsilon')
    parser.add_argument('--alpha', default=0.007, type=float, help='PGD noise attack alpha')
    parser.add_argument('--num_iter', default=10, type=int, help='PGD noise attack iter')
    parser.add_argument('--attack_mode', default=None, type=str, help='attack mode')
    args = parser.parse_args()

    print("*** pgd/fgsm attack script ***")
    print("*** arguments: ***")
    print(args)
    print()

    # initialize scenario
    model: Module = model_selector.get_default_resnet(layers=args.layers, pretrain=args.clean_model)
    train_set: Dataset = dataset.get_cifar10_dataset(True, download=args.load_data, transform='random')
    test_set: Dataset = dataset.get_cifar10_dataset(False, download=args.load_data)
    scenario: Scenario = visualizeScenario(load_path=args.load_path,
                                           save_path=args.save_path,
                                           batch_size=args.batch_size,
                                           lr=args.lr,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay,
                                           test_val_ratio=args.test_val_ratio,
                                           epsilon=args.epsilon,
                                           alpha=args.alpha,
                                           num_iter=args.num_iter,
                                           model=model,
                                           train_set=train_set,
                                           test_set=test_set,
                                           attack_mode=args.attack_mode)

    if not args.dry_run:
        if args.test_only:
            scenario.perform(0)
        else:
            scenario.perform(args.train_epochs)
