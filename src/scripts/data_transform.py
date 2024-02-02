# src/scripts/data_transform.py

"""
This script preprocesses raw SEAO (Système électronique d'appel d'offres) data to extract relevant features 
and targets for further analysis, focusing on open public auctions in three sectors: supply, services, and construction.

Input:
'.../data/raw_data.pkl': A pickled DataFrame containing the raw SEAO data after transformation from xml format.

Output:
    '../../data/bids.npy': bids in CAD
    '../../data/standardized_log_bids.npy': standardized logarithmic bids
    '../../data/average_standardized_log_bids.npy': conditional mean of standardized logarithmic bids (per auction)
    '../../data/var_standardized_log_bids.npy': conditional variance of standardized logarithmic bids (per auction)
    '../../data/data.pkl': dataset including auction features and bids 
    '../../data/features.pkl': auction features only 
    '../../data/transformed_features.npy': transformed auction features (onehot encoding)
    '../../data/transformed_features_squeezed.npy': squeezed (one line per auction) transformed auction features (onehot encoding)
    '../../data/info.pkl': dictionary containing output_info_list, column_transform_info_list, n_bidders and data_dim
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import numpy as np
import os
import sys
import argparse
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
    # Remove weak category signals
    threshold = 5
    for column in df.columns:
        if column != 'bids':
            category_counts = df[column].value_counts()
            weak_categories = category_counts[category_counts < threshold].index.tolist()
            df[column] = df[column].replace(weak_categories, '1e5')
    return df

if __name__=="__main__":

    # Parser for test size data
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="Use a small amount of data for test.")
    args = parser.parse_args()

    # Load raw SEAO data
    current_path = os.path.abspath(os.path.dirname(__file__))
    data_file_path = os.path.join(current_path, '..', '..', 'data', 'raw_data.pkl')
    if args.test:
        data = pd.read_pickle(data_file_path)[:10000]
    else:
        data = pd.read_pickle(data_file_path)

    # Clean the data and extract n_bidders
    data = clean_data(data)

    # Isolate features from bids
    bids = pd.to_numeric(data.bids)
    features = data.drop('bids', axis=1)
    
    # Transform bids
    bids = np.array(bids).reshape(-1,1)
    scaler = StandardScaler()
    standardized_log_bids = scaler.fit_transform(np.log(bids))

    # Get mu and sigma for multi-output regtree and svr
    temp = pd.DataFrame(standardized_log_bids)
    temp.index = data.index
    average_standardized_log_bids = temp.groupby(temp.index).mean()
    var_standardized_log_bids = temp.groupby(temp.index).var().fillna(0)
    del temp

    # Transform features
    discrete_columns = list(features.columns)
    transformer = DataTransformer(seed=42) 
    transformer.fit(features, discrete_columns)
    transformed_features = transformer.transform(features)
    
    # Remove duplicates in features
    compact_size = np.unique(features.index).shape[0]
    transformed_features_squeezed = pd.DataFrame(transformed_features)
    transformed_features_squeezed['index'] = list(data.index)
    transformed_features_squeezed.drop_duplicates(subset='index', keep='first', inplace=True)
    assert transformed_features_squeezed.shape[0] == compact_size, "Wrong size in dimension 0"
    transformed_features_squeezed.drop('index', axis=1, inplace=True)
    transformed_features_squeezed = np.array(transformed_features_squeezed)

    # Info
    info = {
        'output_info_list': transformer.output_info_list,
        'column_transform_info_list': transformer.column_transform_info_list,
        'data_dim': transformer.output_dimensions,
    }

    # Save data
    bids_path = os.path.join(current_path, '../../data/bids.npy')
    np.save(bids_path, bids)
    standardized_log_bids_path = os.path.join(current_path, '../../data/standardized_log_bids.npy')
    np.save(standardized_log_bids_path, standardized_log_bids)
    average_standardized_log_bids_path = os.path.join(current_path, '../../data/average_standardized_log_bids.npy')
    np.save(average_standardized_log_bids_path, average_standardized_log_bids)
    var_standardized_log_bids_path = os.path.join(current_path, '../../data/var_standardized_log_bids.npy')
    np.save(var_standardized_log_bids_path, var_standardized_log_bids)

    data_path = os.path.join(current_path, '../../data/data.pkl')
    data.to_pickle(data_path)
    features_path = os.path.join(current_path, '../../data/features.pkl')
    features.to_pickle(features_path)
    transformed_features_path = os.path.join(current_path, '../../data/transformed_features.npy')
    np.save(transformed_features_path, transformed_features)
    transformed_features_squeezed_path = os.path.join(current_path, '../../data/transformed_features_squeezed.npy')
    np.save(transformed_features_squeezed_path, transformed_features_squeezed)

    info_path = os.path.join(current_path, '../../data/info.pkl')
    with open(info_path, "wb") as f:
        pickle.dump(info, f)