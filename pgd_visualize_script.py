import os
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, Subset, DataLoader

from attack.pgd import PGD
from dataset import dataset
from model import model_selector


def save_tensor_as_img(tensor: Tensor, file_path: str):
    transforms.ToPILImage()(tensor).save(file_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Visualization Script')
    # model parameter
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    parser.add_argument('--pretrain_model', default=True, action='store_false',
                        help='load online pretrained parameters')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
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

    model: Module = model_selector.get_default_resnet(layers=args.layers, pretrain=args.pretrain_model)

    data_set: Dataset = dataset.get_cifar10_dataset(is_train=args.is_train, download=args.load_data)

    assert args.image_index_start >= 0
    assert args.image_index_end <= len(data_set)

    subset_indices = [i for i in range(args.image_index_start, args.image_index_end)]

    subset = Subset(data_set, subset_indices)

    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    attacker: Module = PGD(model, args.epsilon, args.alpha, args.noise_epochs)

    augmented_path = os.path.join("img", args.img_path)
    img_dir: str = os.path.dirname(augmented_path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for (inputs, labels) in dataloader:
        blurred_tensor: Tensor = attacker(inputs, labels)
        noise_tensor: Tensor = (blurred_tensor - inputs) / args.alpha
        prediction: Tensor = model(blurred_tensor)

        for _ in range(inputs.shape[0]):
            true_label: str = dataset.get_cifar10_targets()[labels[_].item()]
            pred_label: str = dataset.get_cifar10_targets()[torch.argmax(prediction[_]).item()]
            save_tensor_as_img(inputs[_], 'img/ori' + str(_) + '_' + true_label + '.jpg')
            save_tensor_as_img(blurred_tensor[_], 'img/blurred' + str(_) + '_' + pred_label + '.jpg')
            save_tensor_as_img(noise_tensor[_], 'img/noise' + str(_) + '.jpg')
