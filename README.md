# baseline-model

## Introduction

This is a group project of HKU STAT7008.
Topic: A Study of Adversarial Attack

An adversarial attack refers to a method used to manipulate input data with the goal of misleading machine learning models. These manipulations, known as perturbations, are carefully crafted to be undetectable by humans, but they can cause the model to produce incorrect outputs. 

To counter these attacks, various techniques have been developed to defend against adversarial attacks, such as adversarial training. 

In this project, our focus will be on studying both adversarial attacks and defence mechanisms using the CIFAR-10 dataset as our benchmark dataset and ResNet-50 serves as our base model.


## Code Structure

We divided the code into 5 modules. They serve the below purposes:
- train_test_load.py: Training and validate base model
- fgsm_attack.py or pgd_attack.py: Attacking the trained model
- xxxxxx.py: Use defence mechanisms to train a robust model
- pdg_visualize_script.py: Visualization for images and graphs
- unit_test.py: Test cases to perform unit tests


Submodules:
- dataset: To load CIFAR-10 dataset
- model: Get PyTorch default model or implement custom ResNet
- scenario: a generalized script which done the data initialization, model parameter loading and exact procedure.

There are 2 folders created and not included in source code:
- data, the folder to store CIFAR10 data
- checkpoint, the folder to store model checkpoints.

## Technical Description

To train a base module, run the script `train_test_load.py`, with options:
- test_only: only run test
- trains_epochs: number of epochs
- load_path: checkpoint file to load under `checkpoint`
- save_path: path with filename to save checkpoint based on best validate score
- load_data: download dataset if not exists

python ./baseline_model/train_test_load.py --load_data --dry_run

python ./baseline_model/train_test_load.py --train_epochs=20 --save_path="resnet18/best.pth"

python ./baseline_model/train_test_load.py --load_path="resnet18/best.pth" --test_only

## Documentation

Follow PEP coding format
For docstring, use [Sphinx](https://www.sphinx-doc.org/en/master/)