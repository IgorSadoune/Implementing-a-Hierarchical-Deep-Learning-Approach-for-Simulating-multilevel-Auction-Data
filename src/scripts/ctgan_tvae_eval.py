#src/scripts/ctgan_tvae_eval.py

"""
PRODUCES RESULTS OF TABLE 2
---------------------------

This script trains three classifiers (K-NN, classification Tree and classification MLP) on synthetic data, 
and evaluates them on real data, in order to generate inception scores.

Inputs:
    ../../data/transformed_features_squeezed.npy
    '../../data/synthetic_data_ctgan.npy'
    '../../data/synthetic_data_tvae.npy'
    ../../data/info.pkl

Outputs:

    inception metrics: three classification reports.
"""

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from transformer import DataTransformer # necessary to load info dict

def get_label_binary(array):
	return [0 if i<0.5 else 1 for i in array]

def data_prep(real_data, synthetic_data, output_info_list, variable_index):
    """
    Prepares data for classification by extracting a target variable and removing its corresponding columns.

    Parameters:
        real_data (np.ndarray): The real data to prepare.
        synthetic_data (np.ndarray): The synthetic data to prepare.
        output_info_list (list): A list of information about the output variables.
        variable_index (int): The index of the target variable in the output_info_list.

    Returns:
        train_data (np.ndarray): The prepared synthetic data.
        test_data (np.ndarray): The prepared real data.
        train_target (np.array): The target variable extracted from the synthetic data.
        test_target (np.array): The target variable extracted from the real data.
    """
    # Calculate the start and end indices for slicing
    start_idx = sum(info[0].dim for info in output_info_list[:variable_index])
    end_idx = start_idx + output_info_list[variable_index][0].dim
    # Slice the array to extract the columns corresponding to the variable at variable_index
    train_target = synthetic_data[:, start_idx:end_idx]
    test_target = real_data[:, start_idx:end_idx]
    # Merge the sliced columns into one binary column
    train_target = np.argmax(train_target, axis=1)
    test_target = np.argmax(test_target, axis=1)
    # Remove these columns from the original array
    train_data = np.delete(synthetic_data, np.s_[start_idx:end_idx], axis=1)
    test_data = np.delete(real_data, np.s_[start_idx:end_idx], axis=1)
    return  train_data, test_data, train_target, test_target 

def inception_score(train_data, test_data, train_target, test_target):
    """
    Trains and evaluates several classifiers on the prepared data, and prints classification reports.

    Parameters:
        train_data (np.ndarray): The synthetic data to train the classifiers on.
        test_data (np.ndarray): The real data to test the classifiers on.
        train_target (np.array): The target variable for the synthetic data.
        test_target (np.array): The target variable for the real data.
    """
    # Init classifiers
    names = [
        'Nearest Neighbors',
        'DecisionTree',
        'MLP'
        ]
    classifiers = [
        KNeighborsClassifier(5), # deterministic
        DecisionTreeClassifier(max_depth=20, random_state=42),
        MLPClassifier(alpha=1, max_iter=1000, early_stopping=True, random_state=42)
        ]
    # Training classifiers and classification reports
    for name, clf in zip(names, classifiers):
        print('---------------------------------------------')
        print(name)
        print('---------------------------------------------')
        clf.fit(train_data, train_target)
        pred_val = clf.predict(train_data)
        pred_test = clf.predict(test_data)
        print('VAL')
        print(classification_report(train_target, get_label_binary(pred_val)))
        print('TEST')
        print(classification_report(test_target, get_label_binary(pred_test)))

if __name__=="__main__":

    # Paths
    current_path = os.path.dirname(os.path.abspath(__file__))
    features_squeezed_path = os.path.join(current_path, '../../data/transformed_features_squeezed.npy')
    synthetic_data_ctgan_path = os.path.join(current_path, '../../data/synthetic_data_ctgan.npy')
    synthetic_data_tvae_path = os.path.join(current_path, '../../data/synthetic_data_tvae.npy')
    info_path = os.path.join(current_path, '../../data/info.pkl')

    # Load info
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    output_info_list = info["output_info_list"]

    # Load synthetic data
    synthetic_data_ctgan = np.load(synthetic_data_ctgan_path)
    synthetic_data_tvae = np.load(synthetic_data_tvae_path)

    # Load real data
    features_squeezed = np.load(features_squeezed_path)

    # Extract binary municipality as target + inception score
    variable_index = 1 #municipality
    for synth, t in zip([synthetic_data_ctgan, synthetic_data_tvae], ['a', 'b']):
        train_data, test_data, train_target, test_target  = data_prep(features_squeezed,
                                                                    synth, 
                                                                    output_info_list, 
                                                                    variable_index)
        print(t)
        print(train_target.shape)
        inception_score(train_data, test_data, train_target, test_target)