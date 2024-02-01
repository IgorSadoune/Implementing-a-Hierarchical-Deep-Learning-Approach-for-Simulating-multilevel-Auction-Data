#src/scripts/bidnet_eval.py

from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
from scipy.stats import norm
import numpy as np
import os
import collections
from math import log2

def flatten(
	listOfElems
	) -> object:
	'''
	Flatten multi-level nested iterables. Use list(flatten(yourlist)).
	
	Output:
		A flat object (use list(object) in order to transcript in a list)
	'''

	for _ in listOfElems:
		if isinstance(
			_,
			collections.Iterable
			) and not isinstance(
			_,
			(str, bytes)
			):
			yield from flatten(_)
		else:
			yield _

def kl_divergence(p, q):
    '''
    '''
    return sum(p[i] * log2((p[i]/q[i]) + 1e-5) for i in range(len(p)))

def get_metrics(pred, target):
	'''
	'''
	if type(target) is list:
		target = np.array(target)
	#get metrics
	rmse = np.sqrt(mean_squared_error(np.sort(target), np.sort(pred)))
	p, q = norm.pdf(target).reshape([-1]), norm.pdf(pred).reshape([-1])
	ws = wasserstein_distance(p, q)
	return rmse, ws

if __name__=="__main__":

	# Load data
	current_path = os.path.dirname(os.path.abspath(__file__))
	real_data_path = os.path.join(current_path, '../../data/transformed_features.npy')
	standardized_log_average_bids_path = os.path.join(current_path, '../../data/average_standardized_log_bids.npy')
	real_data = np.load(real_data_path)
	standardized_log_average_bids = np.load(standardized_log_average_bids_path)

	# Load synthetic bids
	synthetic_bids = {}
	for synthetic_bid in ['b_hat', 'b_tilde_ctgan', 'b_tilde_tvae']:
		synthetic_bids[synthetic_bid] = np.load(os.path.join(current_path, f'../../data/{synthetic_bid}' + '.npy'))
		
	# Metrics
	rmse, ws = {}, {}
	rmse['b_hat_vs_b'], ws['b_hat_vs_b'] = get_metrics(synthetic_bids['b_hat'], standardized_log_average_bids)
	rmse['b_hat_vs_b_tilde_ctgan'], ws['b_hat_vs_btilde_ctgan'] = get_metrics(synthetic_bids['b_hat'], synthetic_bids['b_tilde_ctgan'])
	rmse['b_hat_vs_b_tilde_tvae'], ws['b_hat_vs_btilde_tvae'] = get_metrics(synthetic_bids['b_hat'], synthetic_bids['b_tilde_tvae'])
	rmse['b_tilde_ctagn_vs_b'], ws['b_tilde_ctagn_vs_b'] = get_metrics(synthetic_bids['b_tilde_ctgan'], standardized_log_average_bids)
	rmse['b_tilde_tvae_vs_b'], ws['b_tilde_tvae_vs_b'] = get_metrics(synthetic_bids['b_tilde_tvae'], standardized_log_average_bids)

	# Results
	print('RMSE:', rmse)
	print('WS:', ws)