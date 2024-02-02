@echo off
REM Run data_transform.py: cleans and prepares data for ctgan, tvae and bidnet training
python src\scripts\data_transform.py 

REM Run ctgan_tvae_train.py: trains both DGMs and samples synthetic features from their trained models
python src\scripts\ctgan_tvae_train.py --verbose --save_model

REM Run bidnet_train.py: trains the bidnet to predict bids from features
python src\scripts\bidnet_train.py --verbose --save_model

REM Run msvr.py: trains the MSVR to predict bids from features
python src\scripts\msvr.py --verbose --save_model

REM Run regtree.py: trains the multi-output regression tree to predict bids from features
python src\scripts\regtree.py --verbose --save_model

REM Run ctgan_tvae_eval.py: performs inception scoring by training three classifiers on the real and synthetic data
python src\scripts\ctgan_tvae_eval.py

REM Run bidnet_eval.py: evaluates the BidNet by computing the distance between real, predicted, and synthetic bids distributions
python src\scripts\bidnet_eval.py
