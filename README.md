# baseline-model

## Introduction

This is a group project of HKU STAT7008.
Topic: A Study of Adversarial Attack

An adversarial attack refers to a method used to manipulate input data with the goal of misleading machine learning models. These manipulations, known as perturbations, are carefully crafted to be undetectable by humans, but they can cause the model to produce incorrect outputs. 

To counter these attacks, various techniques have been developed to defend against adversarial attacks, such as adversarial training. 

In this project, our focus will be on studying both adversarial attacks and defence mechanisms using the CIFAR-10 dataset as our benchmark dataset and ResNet-50 serves as our base model.


## Setup Environment

To setup the environment in conda, perform the below commands:

```commandline
# add conda channel to download package
conda config --add channels conda-forge  
# create environment from requirements.txt
conda create --name adtrain_env --file requirements.txt 
```

## Code Structure

We divided the code into 5 modules. They serve the below purposes:
- train_test_load.py: Training and validate base model
- adtrain_script.py: Use defence mechanisms to train a robust model
- kndist_script.py: Use knowledge distillation to train a robust model
- unit_test.py: Test cases to perform unit tests
- integration_test.py: Test case for running actual scenario
- pdg_visualize_script.py: Visualization for images and graphs

Submodules:
- dataset: To load CIFAR-10 dataset
- model: Get PyTorch default model or implement custom ResNet
- attack: implementation for attack like Random Noise, FGSM and PGD
- scenario: a generalized script which done the data initialization, model parameter loading and exact procedure.

There are 2 folders created and not included in source code:
- data, the folder to store CIFAR10 data
- checkpoint, the folder to store model checkpoints.
- img, the folder to store images for visualization purpose

## Technical Description

To train a base module, run the script `train_test_load.py`, with options:
- test_only: only run test
- trains_epochs: number of epochs
- load_path: checkpoint file to load under `checkpoint`
- save_path: path with filename to save checkpoint based on best validate score
- load_data: download dataset if not exists

Example command: 

```python
python train_test_load.py --load_data --dry_run # download dataset and perform dry-run
python train_test_load.py --train_epochs=20 --save_path="resnet18/best.pth" # train and save model
python train_test_load.py --load_path="resnet18/best.pth" --test_only # load and test model 
```

## Documentation

Follow PEP coding format
For docstring, use [Sphinx](https://www.sphinx-doc.org/en/master/)