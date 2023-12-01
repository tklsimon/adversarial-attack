import os
from argparse import ArgumentParser

import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, Subset, DataLoader

from attack.pgd import PGD
from dataset import dataset
from model import model_selector

if __name__ == '__main__':
    parser = ArgumentParser(description='Visualization Script')
    # model parameter
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    parser.add_argument('--clean_model', default=True, action='store_false', help='load online pretrained parameters')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    # train and test parameters
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')
    parser.add_argument('--is_train', default=True, action='store_false', help='is train dataset')
    # attack parameter
    parser.add_argument('--epsilon', default=0.03, type=float, help='PGD noise attack epsilon')
    parser.add_argument('--alpha', default=0.007, type=float, help='PGD noise attack alpha')
    parser.add_argument('--noise_epochs', default=10, type=int, help='no of epochs for PGD noise attack')
    # visualize parameter
    parser.add_argument('--img_path', default="", type=str, help='save image')
    parser.add_argument('--image_index_start', default=0, type=int, help='range of image to be generated')
    parser.add_argument('--image_index_end', default=10, type=int, help='range of image to be generated')
    args = parser.parse_args()

    print("*** visualization script script ***")
    print("*** arguments: ***")
    print(args)
    print()

    model: Module = model_selector.get_default_resnet(layers=args.layers, pretrain=args.clean_model)

    data_set: Dataset = dataset.get_cifar10_dataset(is_train=args.is_train, download=args.load_data)

    assert args.image_index_start >= 0
    assert args.image_index_end <= len(data_set)

    subset_indices = [i for i in range(args.image_index_start, args.image_index_end + 1)]

    subset = Subset(data_set, subset_indices)

    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    augmented_path = os.path.join("img", args.img_path)
    img_dir: str = os.path.dirname(augmented_path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for (inputs, labels) in dataloader:
        _input = inputs[1]
        _label = labels[1]

        pil_image = transforms.ToPILImage()(_input)
        pil_image.save('img/ori.jpg')

        attacker: Module = PGD(model, args.epsilon, args.alpha, args.noise_epochs)
        blurred_tensor: Tensor = attacker(inputs, labels)
        _blurred_tensor = blurred_tensor[1]

        blurred_image = transforms.ToPILImage()(_blurred_tensor)
        blurred_image.save('img/blurred.jpg')

        noise_tensor: Tensor = _blurred_tensor - _input
        noise_tensor /= args.alpha
        noise_image = transforms.ToPILImage()(noise_tensor)
        noise_image.save('img/noise.jpg')

        print("target=", dataset.get_cifar10_targets()[_label.item()])
