import pickle
import pandas as pd
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from transformer import DataTransformer # necessary to load info dict

# Paths
current_path = os.path.dirname(os.path.abspath(__file__))
features_squeezed_path = os.path.join(current_path, '../../data/features_squeezed.npy')
synthetic_data_ctgan_path = os.path.join(current_path, '../../data/synthetic_data_ctgan.npy')
synthetic_data_tvae_path = os.path.join(current_path, '../../data/synthetic_data_tvae.npy')
info_path = os.path.join(current_path, '../../data/info.pkl')

original_features_path = os.path.join(current_path, '../../data/original_features.pkl')
original_features = pd.read_pickle(original_features_path)

# Load info
with open(info_path, 'rb') as f:
    info = pickle.load(f)
output_info_list = info["output_info_list"]
print(sum(info[0].dim for info in output_info_list[:1]))
print(output_info_list[1][0].dim)

print(original_features.columns)