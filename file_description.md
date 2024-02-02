# Table of Contents

1. [EXECUTABLES](#executables)
   - [Root](#root)
   - [src/scripts/](#srcscripts)
     - [data_transform.py](#data_transformpy)
     - [ctgan_tvae_train.py](#ctgan_tvaetrainpy)
     - [ctgan_tvae_eval.py](#ctgan_tvae_evalpy)
     - [bidnet_train.py](#bidnet_trainpy)
     - [msvr.py](#msvrpy)
     - [regtree.py](#regtreepy)
     - [bidnet_eval.py](#bidnet_evalpy)
2. [NON-EXECUTABLES](#non-executables)
   - [src/modules/](#srcmodules)
3. [DATA FILES](#data-files)
4. [TRAINED PARAMETERS](#trained-parameters)
5. [SUPPORT FILES](#support-files)

# Project Documentation

## EXECUTABLES

### Root

- **test_full.sh**: Run all the executables in `src/scripts/` in correct order with a small amount of data. RUNNING THIS TEST FILE WILL OVERRIGHT DATA SAMPLES WITH TEST VERSION IN `data/`, AND THEREFORE WILL NOT PRODUCE THE SAME METRICS. TO EXECUTE BEFORE `full.sh` IF THE USER WANTS TO REPLICATE THE TRAINING PROCESS.

- **full.sh**: Run all the executables in `src/scripts/` in correct order with the entire data, producing the results of Table 2, 3, and 4.

- **validation.sh**: Only run `ctgan_tvae_eval.py` and `bidnet_eval.py`, producing the results of Table 2 and 4. This file allows to replicate the study without training the models, which requires a certain amount of computational resources.

- **.bat files**: FOR WINDOWS USERS, execute `test_full.bat`, `full.bat` or `validation.bat` instead of their `.sh` counterparts.

### src/scripts/

#### data_transform.py

This script preprocesses raw SEAO (Système électronique d'appel d'offres) data to extract relevant features and targets for further analysis, focusing on open public auctions in three sectors: supply, services, and construction.

- **Inputs**:
    - `.../data/raw_data.pkl`: A pickled DataFrame containing the raw SEAO data after transformation from XML format.

- **Outputs**:
    - Various numpy arrays and pickled files containing bids and features.

#### ctgan_tvae_train.py

This script trains two synthetic data generators, CTGAN and TVAE, on the transformed data, and generates synthetic data using the trained models.

- **Inputs/Outputs**: Includes transformed features and model losses.

#### ctgan_tvae_eval.py

PRODUCES RESULTS OF TABLE 2. Trains classifiers on synthetic data and evaluates them on real data.

#### bidnet_train.py

PRODUCES RESULTS OF TABLE 3 FOR THE BidNet. Trains the BidNet and predicts synthetic bids.

#### msvr.py

PRODUCES RESULTS OF TABLE 3 FOR THE MSVR. Trains a multi-output support vector machine regression model.

#### regtree.py

PRODUCES RESULTS OF TABLE 3 FOR THE REGTREE. Trains a multi-output regression tree model.

#### bidnet_eval.py

PRODUCES RESULTS OF TABLE 4. Measures probability distributions distances.

## NON-EXECUTABLES

### src/modules/

Contains custom modules like `transformer.py`, `sampler.py`, `ctgan.py`, `tvae.py`, and `bidnet.py`, each with a specific role in data transformation, sampling, or model customization.

## DATA FILES

Lists various data files like bids, features, synthetic data, and losses with their paths and descriptions.

## TRAINED PARAMETERS

Lists model parameter files like `ctgan_model.pkl`, `tvae_model.pkl`, `bidnet_model.pkl`, `svr_model.pt`, and `regtree_model.pt` with their paths.

## SUPPORT FILES

- **requirements.txt**: Contains a list of Python libraries and packages needed to run the executables.
- **architecture.txt**: Contains the directory structure of the repo. THIS STRUCTURE MUST BE RESPECTED.
