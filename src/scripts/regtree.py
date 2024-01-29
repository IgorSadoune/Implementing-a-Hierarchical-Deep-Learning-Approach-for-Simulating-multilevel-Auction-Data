#src/scripts/regtree.py

import torch
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
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
parser.add_argument('--max_depth', type=int, default=8, help='Tree max depth')
parser.add_argument('--save_model', action='store_true', help='flag to save the model')
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
features = np.load(os.path.join(current_path, '..', '..', 'data', 'features.npy'))
standardized_log_bids = np.load(os.path.join(current_path, '..', '..', 'data', 'standardized_log_bids.npy'))

# Define paths
model_path = os.path.join(current_path, '..', '..', 'models', 'regtree_model.pt')

# Prep data and target
data = torch.tensor(features)
target = torch.tensor(standardized_log_bids)
input_dim = data.shape[1]
dataset = torch.utils.data.TensorDataset(data, target)

# Training loop
val_loss = {}
kfold = KFold(n_splits=args.k_folds, shuffle=True)
best = np.inf
criterion = torch.nn.GaussianNLLLoss()
for fold, (train_ids, val_ids) in tqdm(enumerate(kfold.split(dataset))):

    print(f'FOLD {fold}')
    print('--------------------------------')

    # Init regression tree model
    model = DecisionTreeRegressor(max_depth=args.max_depth)

    # Train/val split 
    train_data, train_target = data[train_ids], target[train_ids]
    val_data, val_target = data[val_ids], target[val_ids]

    # Train
    model.fit(train_data, train_target)

    # Predict
    pred = model.predict(data)
    mu, sigma = torch.tensor(pred[:, 0]).reshape(-1,1), torch.tensor(pred[:, 1]).reshape(-1,1)
    
    # Compute loss
    loss = criterion(mu, torch.tensor(target[:, -1]).reshape(-1,1), sigma)
    val_loss[fold] = loss.item()
    
    print('Validation Loss in Fold', fold, loss.item())

    # Save model
    if loss.item() < best:
        best = loss.item()
        selected = [fold, best]
        if args.save_model:
            pickle.dump(model, open(model_path, 'wb'))
            print('Model saved in Fold', fold)

print(f'K-FOLD CROSS VALIDATION RESULTS FOR {args.k_folds} FOLDS')
print('--------------------------------')
print('Best model: Fold', selected[0], 'validation loss', selected[1])
for key, value in val_loss.items():
    print(f'Best Validation Loss in Fold {key}: {value}')
