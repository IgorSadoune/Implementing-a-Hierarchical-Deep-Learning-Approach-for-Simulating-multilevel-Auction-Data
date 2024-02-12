# Table of Contents

1. [Executables (root)](#executables-root)
2. [Scripts (src/scripts/)](#scripts-srcscripts)
   - [data_transform.py](#data_transformpy)
   - [ctgan_tvae_train.py](#ctgan_tvaetrainpy)
   - [ctgan_tvae_eval.py](#ctgan_tvae_evalpy)
   - [bidnet_train.py](#bidnet_trainpy)
   - [msvr.py](#msvrpy)
   - [regtree.py](#regtreepy)
   - [bidnet_eval.py](#bidnet_evalpy)
   - [qq_plots.py](qq_plots.py)
3. [Modules (src/modules/)](#modules-srcmodules)
   - [transformer.py](#transformerpy)
   - [sampler.py](#samplerpy)
   - [ctgan.py](#ctganpy)
   - [tvae.py](#tvaepy)
   - [bidnet.py](#bidnetpy)
4. [Data Files (data/)](#data-files-data)
5. [TRAINED PARAMETERS](#trained-parameters)
6. [SUPPORT FILES](#support-files)
    - [root](#root)
    - [xml_to_pickle/](#xml-to-pickle/)

# Executables (root)

- **test_full.sh**: Run all the executables in `src/scripts/` in correct order with a small amount of data. To be executed before `full.sh` to prevent errors.
**Running the test routine will overwrite data samples with their test versions in `multi-level-auction-generator/data/`, and therefore will not produce the metrics displayed in the paper.** 

- **full.sh**: Run all the executables in `src/scripts/` in correct order with the entire data, producing the results of Table 2, 3, and 4.

- **validation.sh**: Only run `ctgan_tvae_eval.py` and `bidnet_eval.py`, producing the results of Table 2 and 4. This file allows to replicate the study without training the models, which requires a certain amount of computational resources.

- **.bat files**: FOR WINDOWS USERS, execute `test_full.bat`, `full.bat` or `validation.bat` instead of their `.sh` counterparts.

# Scripts (src/scripts/)

## data_transform.py
Preprocesses raw SEAO data to extract relevant features and targets, focusing on open public auctions in three sectors: supply, services, and construction.
- **Inputs**:
    -data/raw_data.pkl`: A pickled DataFrame containing the raw SEAO data after transformation from XML format.
- **Outputs**:
    - `data/bids.npy`: bids in CAD
    - `data/standardized_log_bids.npy`: standardized logarithmic bids
    - `data/average_standardized_log_bids.npy`: conditional mean of standardized logarithmic bids (per auction)
    - `data/var_standardized_log_bids.npy`: conditional variance of standardized logarithmic bids (per auction)
    - `data/data.pkl`: dataset including auction features and bids 
    - `data/features.pkl`: auction features only 
    - `data/transformed_features.npy`: transformed auction features (onehot encoding)
    - `data/transformed_features_squeezed.npy`: squeezed (one line per auction) transformed auction features (onehot encoding)
    - `data/info.pkl`: dictionary containing output_info_list, column_transform_info_list, n_bidders, and data_dim

## ctgan_tvae_train.py
Trains two synthetic data generators, CTGAN and TVAE, on transformed data, generating synthetic datasets.
- **Inputs**:
    - `data/transformed_features_squeezed.npy`
    - `data/info.pkl`
- **Outputs**:
    - ctgan synthetic data: `data/synthetic_data_ctgan.npy`
    - tvae synthetic data: `data/synthetic_data_tvae.npy`
    - ctgan model: `models/ctgan_model.pkl`
    - tvae model: `models/tvae_model.pkl`
    - ctgan losses: `data/ctgan_losses.pkl`
    - tvae losses: `data/tvae_losses.pkl`

## ctgan_tvae_eval.py
PRODUCES RESULTS OF TABLE 2. This script trains three classifiers (K-NN, classification Tree and classification MLP) on synthetic data, and evaluates them on real data, in order to generate inception scores.
- **Inputs**:
        - `data/transformed_features_squeezed.npy`
        - `data/synthetic_data_ctgan.npy`
        - `data/synthetic_data_tvae.npy`
        - `data/info.pkl`
- **Outputs**:
        - inception metrics: three classification reports

## bidnet_train.py
PRODUCES RESULTS OF TABLE 3 FOR THE BidNet. This script trains the BidNet using K-fold cross-validation and early stopping and uses trained BidNet parameters to predict synthetic bids from real and synthetic features.
-**Inputs**:
    - `data/transformed_features.npy`
    - `data/transformed_features_squeezed.npy`
    - `data/standardized_log_bids.npy`
    - `data/info.pkl`
    - `data/synthetic_data_ctgan.npy`
    - `data/synthetic_data_tvae.npy`
-**Outputs**:
    - Predicted bids from real features: `data/b_hat.npy`
    - Predicted bids from synthetic features (CTGAN): `data/b_tilde_ctgan.npy`
    - Predicted bids from synthetic features (TVAE): `data/b_tilde_tvae.npy`
    - bidnet model: `models/bidnet_model.pkl`
    - bidnet losses: `data/bidnet_losses.pkl`

## msvr.py
 PRODUCES RESULTS OF TABLE 3 FOR THE MSVR. This script trains a multi-output support vector machine regression (MSVR) model.
- **Inputs**:
    - `data/transformed_features_squeezed.npy`
    - `data/average_standardized_log_bids.npy`
    - `data/var_standardized_log_bids.npy`
    - `data/standardized_log_bids.npy`
- **Outputs**:
    - msvr model: `models/svr_model.pt`

## regtree.py
PRODUCES RESULTS OF TABLE 3 FOR THE REGTREE. This script trains a multi-output regression tree (regtree) model.
- **Inputs**:
    - `data/transformed_features_squeezed.npy`
    - `data/average_standardized_log_bids.npy`
    - `data/var_standardized_log_bids.npy`
    - `data/standardized_log_bids.npy`
- **Outputs**:
    - reg tree model: `models/regtree_model.pt`

## bidnet_eval.py
PRODUCES RESULTS OF TABLE 4. This script measures probability distributions distances based on quantile-to-quantile root mean squared error (QQRMSE) and earth mover distance (EMD, aslo called Wasserstein distance).
- **Inputs**:
    - `data/average_standardized_log_bids.npy`
    - `data/b_hat.npy`
    - `data/b_tilde_ctgan.npy`
    - `data/b_tilde_tvae.npy`
- **Outputs**:
    - QQ-RMSE: b_hat vs standardized_log_bids, b_hat vs b_tilde_ctgan, b_hat vs b_tilde_tvae, b_tilde_ctgan vs b, b_tilde_tvae vs b
    - EMD: b_hat vs standardized_log_bids, b_hat vs b_tilde_ctgan, b_hat vs b_tilde_tvae, b_tilde_ctgan vs b, b_tilde_tvae vs b

## qq_plots.py
PRODUCES FIGURE 1. This script outputs the quantile-to-quantile plots of the theoretical normal distribution against mode-specific normalization, minmax normalization and standardization of the logarithmic bids.
- **Inputs**:
    - `../../data/standardized_log_bids.npy`
    - `../../data/bids.npy`
- **Outputs**:
    - `../../data/qq_plots.png`

# Modules (src/modules/)

## transformer.py
DataTransformer module. Modified version of the original CTGAN data_transformer.py file: 
https://github.com/sdv-dev/CTGAN/blob/main/ctgan/data_transformer.py. The main modification 
is the disentenglement of the transformer class form Base Synthesizer and CTGAN and TVAE classes,providing more flexibility.

## sampler.py
DataSampler module. Modified version of the original CTGAN data_sampler.py file: 
https://github.com/sdv-dev/CTGAN/blob/main/ctgan/data_sampler.py.

## ctgan.py
CTGAN module. Customized version of the CTGAN class, inspired by the original CTGAN class implementation: https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py. Modifications involve mainly the structure of the class to best fit the needs of our study, as well as customized random state management.

## tvae.py
TVAE module. Customized version of the CTGAN class, inspired by the original CTGAN class implementation: https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py. Modifications involve mainly the structure of the class to best fit the needs of our study, as well as customized random state management.

## bidnet.py
BidNet network and class agent.

# Data Files (data/)
Data and models (trained parameters) available at https://zenodo.org/records/10649028.

    FILE NAME                           FILE PATH                                FILE DESCRIPTION
    ---------                           ---------                                ----------------
    - bids.npy                          - data/bids.npy                          - bids in CAD
    - standardized_log_bids.npy         - data/standardized_log_bids.npy         - Standardized logarithmic bids
    - average_standardized_log_bids.npy - data/average_standardized_log_bids.npy - Conditional mean of standardized logarithmic bids (per auction)
    - var_standardized_log_bids.npy     - data/var_standardized_log_bids.npy     - Conditional variance of standardized logarithmic bids (per auction)
    - data.pkl                          - data/data.pkl                          - Dataset including auction features and bids
    - features.pkl                      - data/features.pkl                      - Auction features only
    - transformed_features.npy          - data/transformed_features.npy          - Transformed auction features (onehot encoding)
    - transformed_features_squeezed.npy - data/transformed_features_squeezed.npy - Squeezed (one line per auction) transformed auction features (onehot encoding)
    - info.pkl                          - data/info.pkl                          - Dictionary containing output_info_list, column_transform_info_list, n_bidders, and data_dim
    - synthetic_data_ctgan.npy          - data/synthetic_data_ctgan.npy          - CTGAN synthetic data
    - synthetic_data_tvae.npy           - data/synthetic_data_tvae.npy           - TVAE synthetic data
    - ctgan_losses.pkl                  - data/ctgan_losses.pkl                  - CTGAN losses
    - tvae_losses.pkl                   - data/tvae_losses.pkl                   - TVAE losses
    - bidnet_losses.pkl                 - data/bidnet_losses.pkl                 - BidNet losses
    - b_hat.npy                         - data/b_hat.npy                         - Predicted bids from real features
    - b_tilde_ctgan.npy                 - data/b_tild_ctgan.npy                  - Predicted bids from synthetic features (CTGAN)
    - b_tilde_tvae.npy                  - data/b_tilde_tvae.npy                  - Predicted bids from synthetic features (TVAE)


# TRAINED PARAMETERS
Data and models (trained parameters) available at https://zenodo.org/records/10649028.

    FILE NAME            FILE PATH
    ---------            ---------
    - ctgan_model.pkl    - models/ctgan_model.pkl
    - tvae_model.pkl     - models/tvae_model.pkl
    - bidnet_model.pkl   - models/bidnet_model.pkl
    - svr_model.pt       - models/svr_model.pt
    - regtree_model.pt   - models/regtree_model.pt

# SUPPORT FILES

## root

- **requirements.txt**: Contains a list of Python libraries and packages needed to run the executables.

## xml_to_pickle/

**Those files are not use to replicate the study, but can be valuable for users willing to deal with SEAO data files directly.**

- **xml_to_pkl.py**: Class module to convert xml tree into pandas dataframe.
- **xml_unfold.py**: Script converting SEAO xml files into pandas dataframes.
