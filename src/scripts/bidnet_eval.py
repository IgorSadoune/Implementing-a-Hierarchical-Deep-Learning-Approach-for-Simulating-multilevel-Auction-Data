#src/scripts/bidnet_eval.py

"""
PRODUCES RESULTS OF TABLE 4
---------------------------

This script measures probability distributions distances based on quantile-to-quantile root mean squared error (QQRMSE) and 
earth mover distance (EMD, aslo called Wasserstein distance).

Inputs:
    '../../data/average_standardized_log_bids.npy'
    '../../data/b_hat.npy'
    '../../data/b_tilde_ctgan.npy'
    '../../data/b_tilde_tvae.npy'

Outputs:
	QQ-RMSE: b_hat vs standardized_log_bids, b_hat vs b_tilde_ctgan, b_hat vs b_tilde_tvae, b_tilde_ctgan vs b, b_tilde_tvae vs b
	EMD: b_hat vs standardized_log_bids, b_hat vs b_tilde_ctgan, b_hat vs b_tilde_tvae, b_tilde_ctgan vs b, b_tilde_tvae vs b
"""

from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
from scipy.stats import norm
import numpy as np
import os

def get_metrics(pred, target):
	'''
	Computes QQ-RMSE and EMD metrics between prediction and target.
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
	average_standardized_log_bids_path = os.path.join(current_path, '../../data/average_standardized_log_bids.npy')
	average_standardized_log_bids = np.load(average_standardized_log_bids_path)

	# Load synthetic bids
	synthetic_bids = {}
	for synthetic_bid in ['b_hat', 'b_tilde_ctgan', 'b_tilde_tvae']:
		synthetic_bids[synthetic_bid] = np.load(os.path.join(current_path, f'../../data/{synthetic_bid}' + '.npy'))
		
	# Metrics
	rmse, ws = {}, {}
	rmse['b_hat_vs_b'], ws['b_hat_vs_b'] = get_metrics(synthetic_bids['b_hat'], average_standardized_log_bids)
	rmse['b_hat_vs_b_tilde_ctgan'], ws['b_hat_vs_btilde_ctgan'] = get_metrics(synthetic_bids['b_hat'], synthetic_bids['b_tilde_ctgan'])
	rmse['b_hat_vs_b_tilde_tvae'], ws['b_hat_vs_btilde_tvae'] = get_metrics(synthetic_bids['b_hat'], synthetic_bids['b_tilde_tvae'])
	rmse['b_tilde_ctagn_vs_b'], ws['b_tilde_ctagn_vs_b'] = get_metrics(synthetic_bids['b_tilde_ctgan'], average_standardized_log_bids)
	rmse['b_tilde_tvae_vs_b'], ws['b_tilde_tvae_vs_b'] = get_metrics(synthetic_bids['b_tilde_tvae'], average_standardized_log_bids)

	# Results
	print('RMSE:', rmse)
	print('WS:', ws)