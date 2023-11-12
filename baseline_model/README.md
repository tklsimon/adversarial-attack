# baseline-model

## Introduction

This module focuses on train, test and evaluate model. It serves the below purposes:

- Train, Evaluate, Test and Save model
- Add noise before Test

To use this module, run the script `train_test_load.py`.  
Options:

- test_only: only run test
- trains_epochs: number of epochs
- load_path: checkpoint file to load under `checkpoint`
- save_path: path with filename to save checkpoint based on best evaluation score
- load_data: download dataset if not exists

## Technical Description

there are several modules organized as below:

Script (Top Level)

Submodules:

- scenario: a generalized script which done the data initialization, model parameter loading and exact procedure.

The most important method is train_eval_test_save, which does the steps (1) train, (2) evaluate after each train
epoch, (3) test, and (4) save model. Note that each part is optional except the Test step.   

- dataset: a module which load dataset
- model: the customized model structure, and a `model_selector` to render the model 

There are 2 folders created and not included in source code: 

- data, the folder to store CIFAR10 data
- checkpoint, the folder to store model checkpoints.

