from argparse import ArgumentParser

import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, Subset, DataLoader

from baseline_model.dataset import dataset
from baseline_model.model import model_selector
from scenario.pgd_attack_scenario import pgd_attack

if __name__ == '__main__':
    parser = ArgumentParser(description='Visualization Script')
    # model parameter
    parser.add_argument('--layers', default=18, type=int, help='no. of layers in model')
    parser.add_argument('--model_type', default='', type=str, help='custom or default model')
    parser.add_argument('--clean_model', default=True, action='store_false', help='load online pretrained parameters')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    # train and test parameters
    parser.add_argument('--load_path', default=None, type=str, help='load from checkpoint')
    parser.add_argument('--load_data', default=False, action='store_true', help='download data if not available')
    parser.add_argument('--is_train', default=True, action='store_false', help='is train dataset')
    # attack parameter
    parser.add_argument('--epsilon', default=0.03, type=float, help='PGD noise attack epsilon')
    parser.add_argument('--alpha', default=0.007, type=float, help='PGD noise attack alpha')
    parser.add_argument('--noise_epochs', default=10, type=int, help='no of epochs for PGD noise attack')
    # visualize parameter
    parser.add_argument('--img_path', default=None, type=str, help='save image')
    parser.add_argument('--image_index_start', default=0, type=int, help='range of image to be generated')
    parser.add_argument('--image_index_end', default=2, type=int, help='range of image to be generated')
    args = parser.parse_args()

    print("*** visualization script script ***")
    print("*** arguments: ***")
    print(args)
    print()

    if args.model_type == 'custom':
        model: Module = model_selector.get_custom_resnet(layers=args.layers)
    else:
        model: Module = model_selector.get_default_resnet(layers=args.layers, pretrain=args.clean_model)

    data_set: Dataset = dataset.get_default_cifar10_dataset(is_train=args.is_train, download=args.load_data)

    assert args.image_index_start >= 0
    assert args.image_index_end <= len(data_set)
    assert args.load_path is not None or args.load_path != ""

    subset_indices = [i for i in range(args.image_index_start, args.image_index_end + 1)]

    subset = Subset(data_set, subset_indices)

    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    for (inputs, labels) in dataloader:
        pil_image = transforms.ToPILImage()(inputs)

        torchvision.utils.save_image(pil_image, 'test1.png')
        # ori_image = transforms.sdfsdf
        # bluerred_image: Tensor = pgd_attack(model, x, y, args.noise_epochs, args.epsilon, args.alpha)
