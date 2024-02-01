# src/scripts/training_bidnet.py

"""
This script 
    1) trains the BidNet using K-fold cross-validation and early stopping
    2) uses trained BidNet parameters to predict synthetic bids from real and synthetic features

Inputs:
    ../../data/transformed_features.npy
    ../../data/transformed_features_squeezed.npy
    ../../data/standardized_log_bids.npy
    ../../data/info.pkl
    '../../data/synthetic_data_ctgan.npy'
    '../../data/synthetic_data_tvae.npy'

Outputs:
        data:
            'b_hat': Predicted bids from real features 
            'b_tilde_ctgan': Predicted bids from synthetic features (CTGAN)
            'b_tilde_tvae': Predicted bids from synthetic features (TVAE)

        models:
            bidnet model: '../../models/bidnet_model.pkl'
            
        losses:
            bidnet losses: '../../data/bidnet_losses.pkl'
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from bidnet import BidNetAgent

def generate_bids_from_prediction(mu, sigma, seed=None):
    if seed is not None:
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)
        else:
            torch.manual_seed(args.seed)
    # Generate random samples from a normal distribution for each element in mu and sigma
    samples = [torch.normal(mu_i.cpu().detach(), sigma_i.cpu().detach()).cpu().numpy() for mu_i, sigma_i in zip(mu, sigma)]
    # Flatten the list of samples and convert it to a NumPy array
    bid_array = np.array([item for sublist in samples for item in sublist])
    # Reshape the array to have a single column (-1 means the number of rows will be inferred)
    bid_array = bid_array.reshape(-1, 1)
    return bid_array

if __name__=="__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Bidnet Hyperparameters")
    parser.add_argument("--hidden_dim", type=int, default=[256], nargs ='+', help="Hidden layer dimension. Can be augmented with several additional layers.")
    parser.add_argument("--xavier", type=bool, default=True, help="Use Xavier initialization")
    parser.add_argument("--normalize", type=bool, default=False, help="Normalize input data")
    parser.add_argument("--dropout", type=bool, default=False, help="Use dropout")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_epoch", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--min_epoch", type=int, default=8, help="Minimum number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    parser.add_argument("--save_model", action="store_true", help="Save model parameters in 'models' folder.")
    parser.add_argument("--cuda", type=bool, default=True, help="Use GPU for training")
    parser.add_argument("--seed", type=int, default=42, help="Use GPU for training")
    args = parser.parse_args()

    # Load data
    current_path = os.path.dirname(os.path.abspath(__file__))
    features = np.load(os.path.join(current_path, '..', '..', 'data', 'transformed_features.npy'))
    features_squeezed = np.load(os.path.join(current_path, '..', '..', 'data', 'transformed_features_squeezed.npy'))
    standardized_log_bids = np.load(os.path.join(current_path, '..', '..', 'data', 'standardized_log_bids.npy'))
    info = pd.read_pickle(os.path.join(current_path, '..', '..', 'data', 'info.pkl'))
    data_dim = info['data_dim']

    # Define paths
    model_path = os.path.join(current_path, '..', '..', 'models', 'bidnet_model.pt')
    losses_path = os.path.join(current_path, '..', '..', 'data', 'bidnet_losses.pkl')

    # Init Bidnet Class
    model = BidNetAgent(
                    input_dim=data_dim,
                    hidden_dim=args.hidden_dim,
                    xavier=args.xavier,
                    normalize=args.normalize,
                    dropout=args.dropout,
                    k_folds=args.k_folds,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    max_epoch=args.max_epoch,
                    min_epoch=args.min_epoch,
                    patience=args.patience,
                    verbose=args.verbose,
                    cuda=args.cuda,
                    model_path=model_path,
                    seed=args.seed,
                    save_model=args.save_model
                    )

    # Train EmbeddedNet or Net and save model
    model.fit(features, standardized_log_bids)

    # Save losses
    if args.save_model:
        model.save_losses(losses_path)

    # Generate synthetic bids
    synthetic_data_ctgan_path = os.path.join(current_path, '../../data/synthetic_data_ctgan.npy')
    synthetic_data_tvae_path = os.path.join(current_path, '../../data/synthetic_data_tvae.npy')
    synthetic_data_ctgan = np.load(synthetic_data_ctgan_path)
    synthetic_data_tvae = np.load(synthetic_data_tvae_path)

    for data, path in zip(
                    [features_squeezed, synthetic_data_ctgan, synthetic_data_tvae],
                    ['b_hat', 'b_tilde_ctgan', 'b_tilde_tvae']
                    ):
        mu, sigma = model.predict(data)
        pred = generate_bids_from_prediction(mu, sigma, seed=args.seed)
        if args.save_model:
            np.save(os.path.join(current_path, '..', '..', 'data', path + '.npy'), pred)