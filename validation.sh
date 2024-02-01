#!/bin/bash

# Run ctgan_tvae_eval.py: performs inception scoring by training three classifiers on the real and synthetic data
python3 src/scripts/ctgan_tvae_eval.py

# Run bidnet_eval.py: evaluates the BidNet by computing the distance between real, predicted and synthetic bids distributions
python3 src/scripts/bidnet_eval.py
