# src/scripts/msvr.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse 
import os 

def early_stopping(epoch, losses, min_epoch, patience):
	'''
     Regularization method to avoid overfitting
	'''
	if min_epoch is None: 
		min_epoch = (-1 * patience) + 2

	if epoch > min_epoch:
		return losses[-1] > np.mean(losses[patience - 1 : -1])
	else:
		return False

def reset_weights(m):
	'''
	Reset model weights to avoid weight leakage
	'''
	for layer in m.children():
		if hasattr(layer, 'reset_parameters'):
			print(f'Reset trainable parameters of layer = {layer}')
			layer.reset_parameters()

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def MultiHingeLoss(target, pred, model):
    '''
    Multi output Hinge loss with L2 penalty
    '''
    a = torch.sum(torch.clamp(1 - torch.linalg.det(torch.matmul(target.t(), pred)), min=0))
    b = torch.sum(model.model.weight ** 2)  # L2 penalty
    return (a + b)

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

class MSVR(torch.nn.Module):
    '''
    Multi-output support vector regression
    '''

    def __init__(self, input_dim, seed=None):
        super(MSVR, self).__init__()
        self.input_dim = input_dim
        self.model = torch.nn.Linear(self.input_dim, 2)
        self.random_state=seed

    def _set_random_state(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)  # For multi-GPU setups
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _outlayer(self, x):
        dim_ = len(x.size())
        mu, sigma = torch.unbind(x, dim=1)
        mu = torch.unsqueeze(mu, dim=1)
        sigma = torch.unsqueeze(sigma, dim=1)
        sigma = torch.nn.functional.relu(sigma)
        return torch.cat((mu, sigma), dim=dim_-1)

    def forward(self, x):
        return self._outlayer(self.model(x))

# Hyperparameteters
parser = argparse.ArgumentParser(description='Hyperparameters for MSVR')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--k_folds', type=int, default=5, help='number of K-folds for cross-validation')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum number of epochs')
parser.add_argument("--min_epoch", type=int, default=5, help="Minimum number of epochs")
parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
parser.add_argument('--batch_size', type=int, default=500, help='batch size for training')
parser.add_argument('--val_batch_size', type=int, default=500, help='batch size for validation')
parser.add_argument('--recording_interval', type=int, default=10, help='interval for recording metrics')
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
model_path = os.path.join(current_path, '..', '..', 'models', 'msvr_model.pt')

# Prep data and target
data = torch.tensor(features).float()
target = torch.tensor(standardized_log_bids).float()
input_dim = data.shape[1]
dataset = torch.utils.data.TensorDataset(data, target)

# Training loop
stops, best_fold, train_loss_fold = {}, {}, {}
kfold = KFold(n_splits=args.k_folds, shuffle=True)
best_global = np.inf
criterion = torch.nn.GaussianNLLLoss()
for fold, (train_ids, val_ids) in tqdm(enumerate(kfold.split(dataset))):

    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample batch
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Train/val split
    train_loader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=args.batch_size, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=args.val_batch_size, sampler=val_subsampler)

    # Init MSVR
    model = MSVR(input_dim).to(device)
    model.apply(reset_weights)
    if fold == 0:
        print('# parameters:', count_parameters(model))

    # Local init
    epoch = 0
    converged = False
    train_loss_, val_loss_, nll_loss = [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    while not converged:

        print('\n Epoch: ', epoch)

        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader, 0):

            data, target = data.to(device), target.to(device)

            # Zero gradient
            optimizer.zero_grad()

            # Forward pass
            pred = model(data)

            # Unbind multi-output
            mu, sigma = torch.unbind(pred, dim=1)
            mu = torch.unsqueeze(mu, dim=1)
            sigma = torch.unsqueeze(sigma, dim=1)

            # Compute loss in batch
            loss = MultiHingeLoss(target, pred, model)
            nll = criterion(mu, target, sigma)
            train_loss += nll.item()

            # Record
            if batch_idx % args.recording_interval == 0:
                train_loss_.append(nll.item())

            # Gradient update
            loss.backward()
            optimizer.step()

        print('Train NLL: {:.4f}'.format(train_loss/(batch_idx+1)))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(val_loader, 0):

                data, target = data.to(device), target.to(device)

                # Forward pass
                pred = model(data)

                # Unbind multi-output
                mu, sigma = torch.unbind(pred, dim=1)
                mu = torch.unsqueeze(mu, dim=1)
                sigma = torch.unsqueeze(sigma, dim=1)

                # Compute validation loss in batch
                loss = MultiHingeLoss(target, pred, model)
                nll = criterion(mu, target, sigma)
                val_loss += nll.item()

        # Record
        val_loss_.append(val_loss/(batch_idx+1))
        
        print('Validation NLL: {:.4f}'.format(val_loss_[-1]))
        
        # Save model
        if val_loss_[-1] < best_global:
            best_global = val_loss_[-1]
            selected = [fold, epoch, best_global]
            if (args.save_model):
                torch.save(model.state_dict(), model_path)
                print('Model Saved, loss: {:.3f}, achieved in fold: {}'.format(best_global, fold))

        # Track best in fold
        best_fold[fold] = np.min(val_loss_)

        # Early stopping
        if epoch > args.max_epoch - 1:
            converged=True
            print('stopping at epoch ', epoch, ' in fold ', fold)
            stops[fold] = epoch
        else:
            converged = early_stopping(epoch, val_loss_, min_epoch=args.min_epoch, patience=args.patience)
            if converged:
                print('stopping at epoch ', epoch, ' in fold ', fold)
                stops[fold] = epoch

        epoch += 1 

    # Record
    train_loss_fold[fold] = train_loss_
    
    print('Process complete for Fold: ', fold)

print(f'K-FOLD CROSS VALIDATION RESULTS FOR {args.k_folds} FOLDS')
print('--------------------------------')
print('Best model: Fold', selected[0], 'Epoch', selected[1], 'validation loss', selected[2])
for key, value in best_fold.items():
    print(f'Best Validation Loss in Fold {key}: {value}')
for key, value in stops.items():
    print(f'Stop in Fold {key}: {value}')