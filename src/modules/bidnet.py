# src/modules/Bidnet.py

"""
Module for the BidNet neural network.
"""

import torch
import pickle
import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import KFold

class BidNet(torch.nn.Module):
	"""
	BidNet is a PyTorch neural network module that can be used to model Gaussian output distributions by applying appropriate output activation functions.

	Attributes:
		model (torch.nn.Sequential): The main neural network model containing the hidden layers.
	"""

	def __init__(
				self,
				input_dim,
				hidden_dim,
				xavier=True,
				normalize=True,
				dropout=True,
				seed=None
				):
		"""
		Creates appropriate output activation for Gaussian network.

		Args:
			input_dim (int): The number of input features.
			hidden_dim (list): A list of hidden layer dimensions.
			xavier (bool): Indicates if the network uses xavier initialization.
			normalize (bool): Indicates if the network uses batch normalization.
			dropout (bool): Indicates if the network uses dropout.
		
		Returns:
			out (torch.Tensor): Output tensor with dimensions (2, 1).
		"""
		
		super(BidNet, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.xavier = xavier
		self.normalize = normalize
		self.dropout = dropout

		# Create the hidden layers
		def block(in_d, out_d, normalize, dropout, xavier, out=False):
			layers = [torch.nn.Linear(in_d, out_d)]
			if xavier:
				torch.nn.init.xavier_uniform_(layers[0].weight)		
				torch.nn.init.zeros_(layers[0].bias)
			if normalize:
				layers.append(torch.nn.BatchNorm1d(out_d, 0.8))
			if not out:
				layers.append(torch.nn.LeakyReLU(0.2))
			if dropout:
				layers.append(torch.nn.Dropout(p=0.2, inplace=True))
			return layers
		
		dim = self.input_dim
		seq = []
		for item in self.hidden_dim:
			seq += block(in_d=dim, out_d=item, xavier=self.xavier, normalize=self.normalize, dropout=self.dropout)
			dim = item
		seq += block(in_d=dim, out_d=2, xavier=False, normalize=False, dropout=False, out=True)
		self.model = torch.nn.Sequential(*seq)

		self.random_state=seed
		self._set_random_state()

	def _set_random_state(self):
		if self.random_state is not None:
			np.random.seed(self.random_state)
			# Set the seed for PyTorch
			torch.manual_seed(self.random_state)
			# CUDA (GPU)
			torch.cuda.manual_seed(self.random_state)
			torch.cuda.manual_seed_all(self.random_state)  # For multi-GPU setups
			# Additional settings for ensuring reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False

	def outlayer(self, x):
		"""
		Creates appropriate output activation for Gaussian network.

		Args:
			x (torch.Tensor): Output from the previous layer with dimensions (prev_layer_in, prev_layer_out).

		Returns:
			out (torch.Tensor): Output tensor with dimensions (2, 1).
		"""

		#get dim of input
		dim_ = len(x.size())

		#separate parameters
		mu, sigma = torch.unbind(x, dim=1)

		#add one dimension to make the right shape
		mu = torch.unsqueeze(mu, dim=1)
		sigma = torch.unsqueeze(sigma, dim=1)

		#relu to sigma bacause variance is positive
		sigma = torch.torch.nn.functional.relu(sigma)
		return torch.cat((mu, sigma), dim=dim_-1)

	def forward(self, input_data):
		"""
		Forward pass of the BidNet module.

		Args:
			input_data (torch.Tensor): Input tensor if embedding is not used. Defaults to None.
		
		Returns:
			torch.Tensor: The output tensor after applying the output layer activation function.
		"""
		
		# Convert to torch.float32
		input_data = input_data.float()

		return self.outlayer(self.model(input_data))

class BidNetAgent(object):
	"""
	BidNet is a PyTorch neural network module that can be used to model Gaussian output distributions by applying appropriate output activation functions.

	Args:
		hidden_dim (list): A list of hidden layer dimensions.
		embedding_sizes (list): A list of tuples containing the number of unique values and the embedding dimension for each categorical feature.
		output_info_list (list): A list of tuples containing the name of the output and the number of unique values for each categorical output.
		n_cont (int): The number of continuous features.
		input_dim (int): The number of input features.
		xavier (bool): Indicates if the network uses xavier initialization.
		normalize (bool): Indicates if the network uses batch normalization.
		dropout (bool): Indicates if the network uses dropout.
		k_folds (int): The number of folds used for cross validation.
		batch_size (int): The batch size used for training.
		learning_rate (float): The learning rate used for training.
		max_epochs (int): The maximum number of epochs used for training.
		min_epochs (int): The minimum number of epochs used for training.
		patience (int): The number of epochs used for early stopping.
		verbose (bool): Indicates if the training progress is printed.
		cuda (bool): Indicates if CUDA is used for training.
		model_path (str): The path to the model file.     

	Methods:
		fit: Fits the model to the training data.
		predict: Predicts the output for the given input.
		get_losses: Returns the losses of the model.
		save_loss: Returns the loss of the model.
		load_model: Loads the model from the given path.
	"""

	def __init__(
				self,
				input_dim,
				hidden_dim=None,
				xavier=True,
				normalize=True,
				dropout=True,
				k_folds=5,
				batch_size=32,
				learning_rate=2e-4,
				max_epoch=100,
				min_epoch=5,
				patience=10,
				verbose=True,
				cuda=True,
				model_path=None,
				seed=None,
				save_model=False
				):

		self.random_state = seed
		self._set_random_state()
		
		self.input_dim=input_dim
		self.hidden_dim=hidden_dim
		self.xavier=xavier
		self.normalize=normalize
		self.dropout=dropout
		self.min_epoch=min_epoch
		self.max_epoch=max_epoch
		self.patience=patience
		self.k_folds=k_folds
		self.batch_size=batch_size
		self.learning_rate=learning_rate
		self.verbose=verbose
		self.model_path=model_path
		self.save_model=save_model

		assert (self.k_folds > 1), "k_folds must be greater than 1."
		assert (self.max_epoch > 0), "max_epochs must be greater than 1."
		assert (self.min_epoch > 0), "min_epochs must be greater than 1."
		assert (self.patience > 0), "patience must be greater than 1."
		assert (self.patience < self.max_epoch), "patience must be less than max_epochs."
		assert (self.min_epoch < self.max_epoch), "min_epochs must be less than max_epochs."
		assert (self.min_epoch < self.patience), "min_epochs must be less than patience."
		assert (batch_size % 2 == 0), "batch_size must be an even number."
		assert (model_path is None or isinstance(model_path, str)), "model_path must be a string."

		self._stop_epoch = {}
		self._best_fold = {}
		self._train_loss_fold = {} 
		self._val_loss_fold = {}
		self._best_global = np.inf
		self._Kfold = KFold(n_splits=self.k_folds)

		if not cuda or not torch.cuda.is_available():
			device = 'cpu'
		elif isinstance(cuda, str):
			device = cuda
		else:
			device = 'cuda'
		self._device = torch.device(device)
		
		self._model = BidNet(input_dim=self.input_dim, 
							hidden_dim=self.hidden_dim, 
							xavier=self.xavier, 
							normalize=self.normalize, 
							dropout=self.dropout,
							seed=self.random_state).to(self._device)

	def _set_random_state(self):
		if self.random_state is not None:
			np.random.seed(self.random_state)
			# Set the seed for PyTorch
			torch.manual_seed(self.random_state)
			# CUDA (GPU)
			torch.cuda.manual_seed(self.random_state)
			torch.cuda.manual_seed_all(self.random_state)  # For multi-GPU setups
			# Additional settings for ensuring reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False

	def _early_stopping(self, epoch, losses, min_epoch, patience):
		"""
		Early stopping is a regularization method that uses validation loss to determine when to stop training.
		"""
		patience = -1*patience
		if min_epoch > -1*patience:
			min_epoch = -1*patience + 1
		if epoch > min_epoch:
			return losses[-1] > np.mean(losses[patience - 1 : -1])
		else:
			return False

	def _reset_weights(self, m):
		"""
		Resetting model weights to avoid weight leakage.
		"""
		for layer in m.children():
			if hasattr(layer, 'reset_parameters'):
				if self.verbose:
					print(f'Reset trainable parameters of layer = {layer}')
				layer.reset_parameters()

	def fit(self, input_data, target_data):
		"""
		Fit the model to the training data.

		Args:
			input_data (np.ndarray): Input data.
			target_data (np.ndarray): Target data.
		"""
		
		assert (isinstance(input_data, np.ndarray)), "input_data must be a numpy array."
		assert (isinstance(target_data, np.ndarray)), "target_data must be a numpy array."
		
		# Troch dataset
		input_data = torch.from_numpy(input_data)
		target_data = torch.from_numpy(target_data)
		dataset = torch.utils.data.TensorDataset(input_data, target_data)
		
		# Loss function
		criterion = torch.nn.GaussianNLLLoss()

		# Main loop
		split = self._Kfold.split(input_data)

		for fold, (train_ids, val_ids) in tqdm(enumerate(split)):

			# Sample elements randomly from a given list of ids, no replacement.
			train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
			val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

			# Data loaders for training and validation, no shuffle.
			train_loader = torch.utils.data.DataLoader(
							dataset, 
							batch_size=self.batch_size, 
							sampler=train_subsampler,
							drop_last=True)
			val_loader = torch.utils.data.DataLoader(
							dataset,
							batch_size=self.batch_size, 
							sampler=val_subsampler,
							drop_last=True)

			# Reset weights and hyperparameters for each training fold
			self._model.apply(self._reset_weights)
			epoch = 0
			converged = False
			train_loss_, val_loss_ = [], []
			optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

			while not converged:

				# Increment epoch
				epoch += 1 
				
				# Training
				train_loss = 0.0
				self._model.train()
				batch_idx = 0
				for batch_idx, (data, target) in enumerate(train_loader, 0):

					# Get data to cuda if possible
					data, target = data.to(self._device), target.to(self._device)

					# Initialize optimizer
					optimizer.zero_grad()

					pred = self._model(input_data=data)
					mu, sigma = torch.unbind(pred, dim=1)
					mu = torch.unsqueeze(mu, dim=1)
					sigma = torch.unsqueeze(sigma, dim=1)

					loss = criterion(mu, target, sigma**2)
					loss.backward()
					optimizer.step()
					train_loss += loss.item()

				# Average training loss per batch                
				train_loss_.append(train_loss/(batch_idx+1))
						
				# Validation
				val_loss = 0.0

				with torch.no_grad():

					self._model.eval()
					for batch_idx, (data, target) in enumerate(val_loader, 0):

						# Get data to cuda if possible
						data, target = data.to(self._device), target.to(self._device)
						pred = self._model(input_data=data)
						loss = criterion(mu, target, sigma)
						val_loss += loss.item()
						
				# Average validation loss per batch
				val_loss_.append(val_loss/(batch_idx+1))

				# Save the best model
				if val_loss_[-1] < self._best_global:
					self._best_global = val_loss_[-1]
					selected = [fold, epoch, self._best_global]
					if (self.save_model):
						torch.save(self._model.state_dict(), self.model_path)

				# Epoch verbose         
				if self.verbose:
					print('\n Epoch: ', epoch)
					print('Negative Log-Likelyhood Loss')
					print('Validation : {:.4f}'.format(val_loss/(batch_idx+1)))
					print('Training   : {:.4f}'.format(train_loss/(batch_idx+1)))
					if self.save_model:
						print('Model Saved, loss: {:.3f}, achieved in fold: {}'.format(self._best_global, fold))

				# Tracking best in fold
				self._best_fold[fold] = np.min(val_loss_)

				# Early stopping
				if epoch > self.max_epoch - 1:#security
					converged=True
					
				else:
					converged = self._early_stopping(epoch, val_loss_, 
														min_epoch=self.min_epoch, 
														patience=self.patience)
				
			# Record stop epoch    
			self._stop_epoch[fold] = epoch

			#record train and validation losses in fold
			self._train_loss_fold[fold] = train_loss_
			self._val_loss_fold[fold] = val_loss_

			# Fold verbose
			if self.verbose:
				print(f'FOLD {fold}')
				print('--------------------------------')
				print('stopping at epoch ', epoch, ' in fold ', fold)
				print('Process complete for Fold: ', fold)

		# Cross-validation recap
		if self.verbose:
			print(f'K-FOLD CROSS VALIDATION RESULTS FOR {self.k_folds} FOLDS')
			print('--------------------------------')
			print('Best model: Fold', selected[0], 'Epoch', selected[1], 'validation loss', selected[2])

			for key, value in self._best_fold.items():
				print(f'Best Validation Loss in Fold (NLL) {key}: {value}')

			for key, value in self._stop_epoch.items():
				print(f'Stop in Fold {key}: in epoch {value}, after {value * (input_data.shape[0]//self.batch_size)} iterations')

	def save_losses(self, losses_path):
			"""
			Save the losses in a pickle file
			
			Args:
				losses_path (str): path to save the losses. Must be a directory.
			"""
			train_losses_path = losses_path + '_train.pkl'
			val_losses_path = losses_path + '_val.pkl'
			pickle.dump(self._train_loss_fold, open(train_losses_path, 'wb'))
			pickle.dump(self._val_loss_fold, open(val_losses_path, 'wb'))

	def get_losses(self):
			"""
			Returns the train and validation losses of each folds 

			Args:
				losses_path (str): path to save the losses. Must be a directory.
			"""
			return self._train_loss_fold, self._val_loss_fold

	def load_model(self, model_path):
		"""
		Load the model from a pickle file
		
		Args:
			model_path (str): path to load the model. Must be a full path that includes filename and extension.
			embedding_sizes (list) (optional): A list of tuples containing the number of unique values and the embedding dimension for each categorical feature.
		"""
		self._model.load_state_dict(torch.load(model_path))
		self._model.eval()

	def predict(self, input_data):
		"""
		Predict the target variable using the trained model.

		Args:
			input_data (np.ndarray): Input data.
		"""
		# Get data to cuda if possible
		input_data = torch.from_numpy(input_data).to(self._device)
		pred = self._model(input_data=input_data)
		mu, sigma = torch.unbind(pred, dim=1)
		mu = torch.unsqueeze(mu, dim=1)
		sigma = torch.unsqueeze(sigma, dim=1)
		return mu, sigma