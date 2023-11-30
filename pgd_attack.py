import os
import sys
from torch.nn import Module
from torch.utils.data import Dataset
from argparse import ArgumentParser

from dataset import dataset
from model import model_selector

par_dir: str = os.path.dirname(os.getcwd())
sys.path.append(par_dir)

from scenario.scenario import Scenario  # noqa
from scenario.adtrain_scenario import AdTrainScenario  # noqa

if __name__ == '__main__':
    print("*** pgd attack script ***")

    # Initialize Parser for Arguments
    parser = ArgumentParser(description='PGD Attack Argument')

    # Model Parameters
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--save_path', default=None, type=str, help='save checkpoint')
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    parser.add_argument('--clean_model', default=True, action='store_false', help='load online pretrained parameters')

    # Train and Test Parameters
    parser.add_argument('--train_epochs', default=10, type=int, help='no. of epochs for train')
    parser.add_argument('--test_val_ratio', default=0.5, type=float, help='ratio for train-eval split')
    parser.add_argument('--test_only', default=False, action='store_true', help='only test model')
    parser.add_argument('--dry_run', default=False, action='store_true', help='will not train or test model')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')

    # Attack Parameter
    parser.add_argument('--epsilon', default=0.03, type=float, help='PGD noise attack epsilon')
    parser.add_argument('--alpha', default=0.007, type=float, help='PGD noise attack alpha')
    parser.add_argument('--noise_epochs', default=10, type=int, help='no of epochs for PGD noise attack')

    args = parser.parse_args()

    print("*** arguments: ***")
    print(args, "/n")

    # Initialize Resnet Model
    model: Module = model_selector.get_default_resnet(layers=args.layers, pretrain=args.clean_model)

    # Load Training Set and Test Set
    train_set: Dataset = dataset.get_cifar10_dataset(True, download=args.load_data)
    test_set: Dataset = dataset.get_cifar10_dataset(False, download=args.load_data)

    # Initialize Scenario
    scenario: Scenario = AdTrainScenario(load_path=args.load_path,
                                         save_path=args.save_path,
                                         batch_size=args.batch_size,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay,
                                         test_val_ratio=args.test_val_ratio,
                                         epsilon=args.epsilon,
                                         alpha=args.alpha,
                                         noise_epochs=args.noise_epochs,
                                         model=model,
                                         train_set=train_set,
                                         test_set=test_set)

    # Perform Training and Testing
    if not args.dry_run:
        if args.test_only:
            print("Will perform testing process only.")
            scenario.perform(0)
        else:
            print("Will perform training and testing process.")
            scenario.perform(args.train_epochs)
    else:
        print("Will perform dry run only.")
        print("Dry run completed.")

    print("Done.")
