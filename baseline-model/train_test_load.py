from argparse import ArgumentParser

from model.baseline_model import BaselineModel
from model.cifar_resnet_model import Resnet18Model

if __name__ == '__main__':
    parser = ArgumentParser(description='PyTorch ResNet CIFAR10 Training')
    # model parameters
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    # train and test parameters
    parser.add_argument('--train_epochs', default=10, type=float, help='no. of epochs for train')
    parser.add_argument('--test_only', default=False, help='only test model')
    parser.add_argument('--dry_run', default=True, help='will not train or test model')
    args = parser.parse_args()

    print("*** test-test-load script ***")
    model: BaselineModel = Resnet18Model(None, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay, resume=args.resume)
    print(model)

    if args.test_only:
        print("===Test Model===")
        if not args.dry_run:
            model.test()
    else:
        for epoch in range(args.train_epochs):
            print()
            print('===Epoch: %d===' % epoch)
            # train
            print("===Train Model===")
            if not args.dry_run:
                model.train()
            # test
            print("===Test Model===")
            if not args.dry_run:
                model.test()
