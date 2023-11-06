from model.resnet_18_model import Resnet18Model
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='PyTorch ResNet CIFAR10 Training')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    args = parser.parse_args()

    print("*** test-test-load script ***")
    model = Resnet18Model(None, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, resume=args.resume)
    print(model)
