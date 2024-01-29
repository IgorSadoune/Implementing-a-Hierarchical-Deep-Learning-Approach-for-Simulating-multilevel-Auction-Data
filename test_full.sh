#!/bin/bash

# Run data_transform.py: cleans and prepares data for ctgan, tvae and bidnet training
python3 src/scripts/data_transform.py

# Run ctgan_tvae_train.py: trains both DGMs and samples synthetic features from their trained models
python3 src/scripts/ctgan_tvae_train.py --ctgan_epochs 2 --ctgan_batch_size 10 --tvae_epochs 2 --tvae_batch_size 10 --verbose

# Run bidnet_train.py: trains the bidnet to predict bids from features
python3 src/scripts/bidnet_train.py --max_epochs 10 --min_epochs 2 --patience 4 --batch_size 10 --verbose

# Run ctgan_tvae_eval.py: performs inception scoring by training three classifiers on the real and synthetic data
python3 src/scripts/ctgan_tvae_eval.py

# Run bidnet_eval.py: evaluates the BidNet by computing the distance between real, predicted and synthetic bids distributions
python3 src/scripts/bidnet_eval.py

