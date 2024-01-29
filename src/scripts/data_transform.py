# src/scripts/data_transform.py

"""
This script preprocesses raw SEAO (Système électronique d'appel d'offres) data to extract relevant features and targets for further analysis, focusing on open public auctions in three sectors: supply, services, and construction.

Input:
'.../data/raw/raw_data.pkl': A pickled DataFrame containing the original raw SEAO data.

Output:
'.../data/processed/data.pkl': A pickled DataFrame containing the preprocessed features and target values (montantsoumis) for each contract.

The script performs the following steps:
- Load the original raw SEAO data.
- Filter the data to keep only open public auctions in supply, services, and construction sectors.
- Drop unnecessary columns from the dataset.
- Create a binary variable 'post_adjudication_expenses' to indicate if there were any expenses after the auction.
- Drop rows with null or NaN values.
- Drop duplicates from the multi-label columns ('nomorganisation' and 'montantsoumis'), and then drop the 'nomorganisation' column.
- Check for inconsistencies in the number of bids and drop rows with such inconsistencies.
- Drop contracts with bids less than 25,000 CAD.
- Rename the columns for better readability.
- Save the preprocessed features and target values as a pickled file.
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modules'))
from transformer import DataTransformer

def clean_data(df):
    '''
    Apply data cleaning steps.
    '''
    # Initial shape    
    initial_shape = df.shape[0]
    print("number of bids:", df.shape[0], "number of contracts:", len(df.index.unique().tolist()))
    # Select open public auctions in 3 sectors (supply, services and construction)
    df = df.loc[df.type=='3']
    df = df.loc[df.nature.isin(['1', '2', '3'])]
    # df = df.loc[df.nature.isin(['3'])] # only construction 
    filtered_shape = df.shape[0]
    print("Rows dropped by filtering type and nature:", initial_shape - filtered_shape)
    print("number of bids:", df.shape[0], "number of contracts:", len(df.index.unique().tolist()))
    # Delete unnecessary columns 
    df = df.drop([
    'numero_x',
    'adresse1_x',
    'adresse2_x',
    'province_x',
    'pays_x',
    'codepostal_x',
    'titre',
    'precision',
    'disposition',
    'hyperlienseao',
    'datesaisieouverture',
    'datesaisieadjudication',
    'adresse1_y',
    'adresse2_y',
    'province_y',
    'pays_y',
    'codepostal_y',
    'neq',
    'montantssoumisunite',
    'nomcontractant_x',
    'neqcontractant_x',
    'numero_y',
    'numero',
    'datepublicationdepense',
    'description',
    'nomcontractant_y',
    'neqcontractant_y',
    'montanttotalcontrat',
    'datedepense',
    'montantcontrat',
    'montantfinal',
    'montantdepense',
    'datefermeture',
    'datefinale',
    'dateadjudication',
    'datepublication',
    'datepublicationfinale',
    'ville_x',
    'conforme',
    'admissible',
    'ville_y',
    'type',
    'adjudicataire'
    ], axis=1)
    # Create binary variable post_auction_expenses
    df['post_auction_expenses'] = df.depenses.fillna('0')
    df['post_auction_expenses'] = np.where(df['post_auction_expenses'] != '0', '1', '0')
    df = df.drop('depenses', axis=1)
    # Find and drop the indexes of null or NaN values
    df = df.drop(df[df.isnull().any(axis=1)].index, axis=0)
    dropped_nulls_shape = df.shape[0]
    print("Rows dropped due to null or NaN values:", filtered_shape - dropped_nulls_shape)
    print("number of bids:", df.shape[0], "number of contracts:", len(df.index.unique().tolist()))
    # Drop inconsistencies regarding the number of bids and keep only competitive auction (number of bidders > 2)
    num_rows = df.groupby(df.index).size()
    indexes_to_remove = num_rows[num_rows != df['fournisseurs'].groupby(df.index).first()].index
    df = df.drop(indexes_to_remove)
    dropped_inconsistencies_shape = df.shape[0]
    print("Rows dropped due to inconsistencies in the number of bids:", dropped_nulls_shape - dropped_inconsistencies_shape)
    print("number of bids:", df.shape[0], "number of contracts:", len(df.index.unique().tolist()))
    # Drop duplicates 
    multi_label_col = [
    'nomorganisation',
    'montantsoumis'
    ]# nomorganisation (firm) is not needed for the analysis but we need it to detect row duplicates in the multi-label space
    df = df.drop_duplicates(subset=multi_label_col, keep='first')
    df = df.drop('nomorganisation', axis=1)
    dropped_duplicates_shape = df.shape[0]
    print("Rows dropped due to duplicates:", dropped_inconsistencies_shape - dropped_duplicates_shape)
    print("number of bids:", df.shape[0], "number of contracts:", len(df.index.unique().tolist()))
    # Drop contracts admitting bids < 25,000 CAD
    df = df.drop(df[pd.to_numeric(df['montantsoumis']) <= 25000].index, axis=0)
    final_shape = df.shape[0]
    print("Rows dropped due to contracts with bids less than 25,000 CAD:", dropped_duplicates_shape - final_shape)
    print("number of bids:", df.shape[0], "number of contracts:", len(df.index.unique().tolist()))
    # Renaming columns
    col_map = {
    'organisme':'public_contractor',
    'municipal':'municipality',
    'nature':'sector',
    'categorieseao':'subsector',
    'regionlivraison':'location',
    'unspscprincipale':'unspsc',
    'fournisseurs':'n_bidders',
    'montantsoumis':'bids'
    }
    df = df.rename(columns=col_map)
    # Isolate bids
    df.bids = pd.to_numeric(df.bids)
    bids = df.bids
    average_bids = df.groupby(df.index)['bids'].mean()
    features = df.drop('bids', axis=1)
    # Remove duplicates in features
    compact_size = np.unique(features.index).shape[0]
    features['index'] = features.index
    features_squeezed = features.drop_duplicates(subset='index')
    features = features.drop('index', axis=1)
    features_squeezed = features_squeezed.drop('index', axis=1)
    assert features_squeezed.shape[0] == compact_size, "Wrong size in dimension 0"
    # Remove weak category signals
    threshold = 5
    for column in features_squeezed.columns:
        category_counts = features_squeezed[column].value_counts()
        weak_categories = category_counts[category_counts < threshold].index.tolist()
        features[column] = features[column].replace(weak_categories, '1e5')
        features_squeezed[column] = features_squeezed[column].replace(weak_categories, '1e5')
    return features, features_squeezed, bids, average_bids

# Load raw SEAO data
current_path = os.path.abspath(os.path.dirname(__file__))
data_file_path = os.path.join(current_path, '..', '..', 'data', 'raw_data.pkl')
data = pd.read_pickle(data_file_path)

# Clean the data and extract n_bidders
features, features_squeezed, bids, average_bids = clean_data(data)
original_features = features
n_bidders = np.array(pd.to_numeric(features_squeezed.n_bidders)).reshape(-1,1)
data_index = list(features.index)

# Transform bids
scaler = StandardScaler()
log_bids = np.array(np.log(bids)).reshape(-1,1)
log_average_bids = np.array(np.log(average_bids)).reshape(-1,1)
standardized_log_bids = scaler.fit_transform(log_bids)
standardized_log_average_bids = scaler.fit_transform(log_average_bids)

# Transform features
discrete_columns = list(features.columns)
transformer = DataTransformer(seed=42) 
transformer.fit(features, discrete_columns)
features = transformer.transform(features)
transformer.fit(features_squeezed, discrete_columns)
features_squeezed = transformer.transform(features_squeezed)

# Info
info = {
    'output_info_list': transformer.output_info_list,
    'column_transform_info_list': transformer.column_transform_info_list,
    'data_dim': transformer.output_dimensions,
    'data_index': data_index,
    'n_bidders': n_bidders
}

# Save data
current_path = os.path.abspath(os.path.dirname(__file__))
original_features_path = os.path.join(current_path, '../../data/original_features.pkl')
features_path = os.path.join(current_path, '../../data/features.npy')
features_squeezed_path = os.path.join(current_path, '../../data/features_squeezed.npy')
features_inception_score_path = os.path.join(current_path, '../../data/features_inception_score.npy')
standardized_log_bids_path = os.path.join(current_path, '../../data/standardized_log_bids.npy')
standardized_log_average_bids_path = os.path.join(current_path, '../../data/standardized_log_average_bids.npy')
info_path = os.path.join(current_path, '../../data/info.pkl')
original_features.to_pickle(original_features_path)
np.save(features_path, features)
np.save(features_squeezed_path, features_squeezed)
np.save(standardized_log_bids_path, standardized_log_bids)
np.save(standardized_log_average_bids_path, standardized_log_average_bids)
with open(info_path, "wb") as f:
    pickle.dump(info, f)