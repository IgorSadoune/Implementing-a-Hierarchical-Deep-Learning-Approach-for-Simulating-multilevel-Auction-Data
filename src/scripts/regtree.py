#src/scripts/regtree.py

"""
This script trains a multi-output regression tree (regtree) model.

Inputs:
    ../../data/transformed_features_squeezed.npy
    ../../data/average_standardized_log_bids.npy
    ../../data/var_standardized_log_bids.npy
    ../../data/standardized_log_bids.npy

Outputs:
        reg tree model: '../../models/regtree_model.pt'
"""

import torch
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
import argparse 
import os 

def set_random_state(seed):
    '''
    Set random state for numpy and torch random processes
    '''
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
# Hyperparameteters
parser = argparse.ArgumentParser(description='Hyperparameters for regression tree')
parser.add_argument('--k_folds', type=int, default=5, help='number of K-folds for cross-validation')
parser.add_argument('--max_depth', type=int, default=10, help='Tree max depth')
parser.add_argument('--save_model', action='store_true', help='flag to save the model')
parser.add_argument("--verbose", action="store_true", help="Display detailed training progress information.")
parser.add_argument("--cuda", type=bool, default=True, help="Use GPU for training")
parser.add_argument("--seed", type=int, default=42, help="Use GPU for training")
args = parser.parse_args()

# Seeding 
set_random_state(args.seed)

# Cuda
torch.cuda.empty_cache() # empty cuda cache
if not args.cuda or not torch.cuda.is_available():
    _device = 'cpu'
elif isinstance(args.cuda, str):
    _device = args.cuda
else:
    _device = 'cuda'
device = torch.device(_device)

# Load data
current_path = os.path.dirname(os.path.abspath(__file__))
features = np.load(os.path.join(current_path, '..', '..', 'data', 'transformed_features_squeezed.npy'))
average_standardized_log_bids = np.load(os.path.join(current_path, '..', '..', 'data', 'average_standardized_log_bids.npy'))
var_standardized_log_bids = np.load(os.path.join(current_path, '..', '..', 'data', 'var_standardized_log_bids.npy'))
standardized_log_bids =  np.load(os.path.join(current_path, '..', '..', 'data', 'standardized_log_bids.npy'))

# Define paths
model_path = os.path.join(current_path, '..', '..', 'models', 'regtree_model.pt')

# Prep data and target
data = torch.tensor(features)
target = np.concatenate((average_standardized_log_bids, var_standardized_log_bids), axis=1)
target = torch.from_numpy(target).float()
nll_target = torch.from_numpy(standardized_log_bids).float()
input_dim = data.shape[1]
dataset = torch.utils.data.TensorDataset(data, target)

# Training loop
val_loss = {}
kfold = KFold(n_splits=args.k_folds, shuffle=True)
best = np.inf
criterion = torch.nn.GaussianNLLLoss()
for fold, (train_ids, val_ids) in tqdm(enumerate(kfold.split(dataset))):
    if args.verbose:
        print(f'FOLD {fold}')
        print('--------------------------------')

    # Init regression tree model
    tree = DecisionTreeRegressor(max_depth=args.max_depth)

    # Wrap it in a MultiOutputRegressor
    model = MultiOutputRegressor(tree)

    # Train/val split 
    train_data, train_target = data[train_ids], target[train_ids]
    val_data, val_target = data[val_ids], nll_target[val_ids]

    # Train
    model.fit(train_data, train_target)

    # Predict
    pred = model.predict(val_data)

    # Unbind multi-output
    mu = torch.from_numpy(pred[:, 0]).float()
    sigma = torch.from_numpy(pred[:, 1]).float()
    
    # Compute loss
    loss = criterion(mu, val_target.reshape(-1,1).detach(), sigma)
    val_loss[fold] = loss.item()
    
    if args.verbose:
        print('Validation Loss in Fold', fold, loss.item())

    # Save model
    if loss.item() < best:
        best = loss.item()
        selected = [fold, best]
        if args.save_model:
            pickle.dump(model, open(model_path, 'wb'))
            if args.verbose:
                print('Model saved in Fold', fold)

if args.verbose:
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {args.k_folds} FOLDS')
    print('--------------------------------')
    print('Best model: Fold', selected[0], 'validation loss', selected[1])
    for key, value in val_loss.items():
        print(f'Best Validation Loss in Fold {key}: {value}')
