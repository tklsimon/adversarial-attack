# baseline-model

This module focuses on train, test and evaluate model.  It serves the below purposes:

- Train, Evaluate, Test and Save model
- Add noise before Test

To use this module, run the script `train_test_load.py`.  
Options:
- test_only: only run test
- trains_epochs: number of epochs
- load_path: checkpoint file to load under `checkpoint`
- save_path: path with filename to save checkpoint based on best evaluation score
- load_data: download dataset if not exists

