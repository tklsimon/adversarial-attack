"""
Import cifar10 image data with PyTorch
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def main():
    print("*** download data script ***")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("===inspect train data===")

    print("target class available: ")
    print(classes)
    print()

    print("===train loader===")
    print("no. of training sample: %d", len(trainloader))
    print("batch size: %d", trainloader.batch_size)
    print()

    print("===train loader===")
    print("no. of test sample: %d", len(testloader))
    print("batch size: %d", testloader.batch_size)
    print()

    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #     print(' '.join(f'{classes[targets[j]]:5s}' for j in range(batch_size)))
    #
    # for batch_idx, (inputs, targets) in enumerate(testloader):
    #     print(' '.join(f'{classes[targets[j]]:5s}' for j in range(batch_size)))


if __name__ == '__main__':
    main()
