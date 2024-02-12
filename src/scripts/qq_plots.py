#src/scripts/qq_plots.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from transformer import DataTransformer
from sklearn.preprocessing import MinMaxScaler
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

"""
PRODUCES FIGURE 1
-----------------

This script outputs the quantile-to-quantile plots of Figure 1.

Inputs:
    '../../data/standardized_log_bids.npy'
    '../../data/bids.npy'

Outputs:
	QQ-plots: '../../data/qq_plots.png'
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Seed for random state management")
args = parser.parse_args()

# For reproducibility
np.random.seed(args.seed)  

# Load data
current_path = os.path.dirname(os.path.abspath(__file__))
bids_path = os.path.join(current_path, '../../data/bids.npy')
bids = np.load(bids_path)
standardized_log_bids_path = os.path.join(current_path, '../../data/standardized_log_bids.npy')
standardized_log_bids = np.load(standardized_log_bids_path).ravel()

log_bids = np.log(bids)

# Minmax log bids
scaler = MinMaxScaler(feature_range=(0, 1))
minmax_log_bids = scaler.fit_transform(log_bids).ravel()

# Mode specific log bids
discrete_columns = []
transformer = DataTransformer(seed=args.seed)
transformer.fit(bids, discrete_columns)
mode_specific_log_bids = transformer.transform(log_bids)[:, 0]

plt.style.use('ggplot')
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# Mode Specific Log Bids QQ-plot
(result, _) = stats.probplot(mode_specific_log_bids, dist="norm", plot=None)[:2]
axs[0].set_title('Mode Specific')
axs[0].set_xlabel('')
axs[0].set_ylabel('Theoretical Quantiles')
axs[0].plot(result[0], result[1], 'o', color='gray')
min_val, max_val = axs[0].get_xlim()
axs[0].plot([min_val, max_val], [min_val, max_val], 'r', label='y=x')

# MinMax Log Bids QQ-plot
(result, _) = stats.probplot(minmax_log_bids, dist="norm", plot=None)[:2]
axs[1].set_title('MinMax')
axs[1].set_xlabel('Logarithmic Bid Distribution')
axs[1].set_ylabel('')
axs[1].plot(result[0], result[1], 'o', color='gray')
min_val, max_val = axs[1].get_xlim()
axs[1].plot([min_val, max_val], [min_val, max_val], 'r')

# Standardized Log Bids QQ-plot
(result, _) = stats.probplot(standardized_log_bids, dist="norm", plot=None)[:2]
axs[2].set_title('Standardized')
axs[2].set_xlabel('')
axs[2].set_ylabel('')

axs[2].plot(result[0], result[1], 'o', color='gray')
min_val, max_val = axs[2].get_xlim()
axs[2].plot([min_val, max_val], [min_val, max_val], 'r')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(current_path, '../../data', 'qq_plots.png')) 

