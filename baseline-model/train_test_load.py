from argparse import ArgumentParser

from torch.nn import Module

from dataset import dataset
from model import model_selector
from scenario.base_train_test_scenario import BaseTrainTestScenario
from scenario.train_test_scenario import TrainTestScenario

if __name__ == '__main__':
    parser = ArgumentParser(description='PyTorch ResNet CIFAR10 Training')
    # model parameters
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    # train and test parameters
    parser.add_argument('--train_epochs', default=10, type=int, help='no. of epochs for train')
    parser.add_argument('--train_eval_ratio', default=0.99, type=float, help='ratio for train-eval split')
    parser.add_argument('--test_only', default=False, action='store_true', help='only test model')
    parser.add_argument('--dry_run', default=False, action='store_true', help='will not train or test model')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')
    # model parameter
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--save_path', default=None, type=str, help='save checkpoint')
    args = parser.parse_args()

    print("*** train-test-load script ***")

    # initialize scenario
    model: Module = model_selector.get_default_resnet(layers=152)
    train_set, test_set = dataset.get_random_cifar10_dataset(args.load_data)
    scenario: BaseTrainTestScenario = TrainTestScenario(load_path=args.load_path,
                                                        save_path=args.save_path,
                                                        batch_size=args.batch_size,
                                                        lr=args.lr,
                                                        momentum=args.momentum,
                                                        weight_decay=args.weight_decay,
                                                        train_eval_ratio=args.train_eval_ratio,
                                                        model=model,
                                                        train_set=train_set,
                                                        test_set=test_set)
    print(scenario)

    if args.test_only:
        scenario.train_eval_test_save(0)
    else:
        scenario.train_eval_test_save(args.train_epochs)
