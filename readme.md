We present a deep learning solution to address the challenges of simulating realistic synthetic first-price sealed-bid auction data. The complexities encountered in this type of auction data include high-cardinality discrete feature spaces and a multilevel structure arising from multiple bids associated with a single auction instance. Our methodology combines deep generative modeling (DGM) with an artificial learner that predicts the conditional bid distribution based on auction characteristics, contributing to advancements in simulation-based research. This approach lays the groundwork for creating realistic auction environments suitable for agent-based learning and modeling applications. Our contribution is twofold: we introduce a comprehensive methodology for simulating multilevel discrete auction data, and we underscore the potential of DGM as a powerful instrument for refining simulation techniques and fostering the development of economic models grounded in generative AI.

# Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - [Clone the Repository](#clone-the-repository)
   - [Virtual Environment (optional but recommended)](#virtual-environment-optional-but-recommended)
   - [Install the Required Dependencies](#install-the-required-dependencies)
3. [Data Download](#data-download)
4. [File Description](#file-description)
5. [Study Replication](#study-replication)
   - [Full Routine](#full-routine)
   - [Validation Routine](#validation-routine)
   - [Run Python Files Individually](#run-python-files-individually)
6. [License](#license)


# Requirements 

- Python3.8 or above 
- pip package installer (usually installed automatically with Python)
- 32GB RAM
- GPU access (optional but recommended)
- Mac OS, Linux distribution or Windows

# Installation

(Via command line)

## Clone the Repository

`git clone https://github.com/IgorSadoune/multi-level-auction-generator.git`

## Virtual Environment (optional but recommended)

1. Create a virtual environment inside the downloaded repository

Go to the root of the folder "multi-level-auction-generator" and execute 

`python3 -m venv venv`

2. Activate the virtual environment 

Then, the virtual environment needs to be activated when you execute files from this repository. 

- On Mac/Linux, execute:
  `source venv/bin/activate`
  
- On Windows, execute:
  `.\venv\Scripts\activate`

## Install the Required Dependencies:

The required python libraries are listed in the "requirements.txt" file. Those can directly be downloaded to your virtual environment (or root system if venv not setup) by executing

`pip install -r requirements.txt`

# Data Download

1. Download the `datasets.zip` file from [this link](https://zenodo.org/records/10649028).

2. Extract `datasets.zip` to a location outside of `multi-level-auction-generator/`.

3. Copy all the 20 files inside the `datasets` folder (the newly unfolded archive) to the root of `multi-level-auction-generator/data/` folder. 
**Make sure to copy and not move the files, as the datasets in the multi-level-auction-generator/data/ folder might be overwritten in subsequent steps.**

5. (optional) Download the `model_parameters.zip`, and extract it into the `multi-level-auction-generator/models/` folder.

# File Description
We recommend reading the `file_description.md` file [here](https://github.com/IgorSadoune/multi-level-auction-generator/blob/master/file_description.md), which catalogs and describes all the files that make up the project.

# Study Replication

**Always place yourself at the root of the repository (multi-level-auction-generator/)**

There are two ways to replicate the study:

Either 

1. You can execute the "Full Routine", which trains and evaluates models, replicating Table 2, 3 and 4. List of model being trained by this routine: CTGAN, TVAE, BIDNET, MSVR, REGTREE and the three classifiers for the inception score.

Or

2. You can follow the "Validation Routine" to produce results of Table 2 and 4 (You need to train the BidNet, MSVR and regtree in order to get the results of Table 3).

## Full Routine

Run the entire routine, including the training of the CTGAN, TVAE, BidNet and classifiers for inception scoring. Note that the full routine took approximately 8 hours on a RTX2060.

1. Make sure that the pickle file `raw_data.pkl` is in the folder `multi-level-auction-generator/data/`. Only this one file is necessary to run the full routine. 

2. Run the test file which execute the routine with a small amount of data and training iterations

- On Mac/Linux, execute:
 `bash test_full.sh`

- For Windows users:
 `test_full.bat`

3. If the test is successful (no error was raised), run the routine using the entirety of the data

- On Mac/Linux, execute:
 `bash full.sh`

- For Windows users:
 `full.bat`

This will train the models and output the results displayed in Table 2, 3, and 4.

## Validation Routine

To replicate the study without training the models

1. Make sure that the files:

- `transformed_features_squeezed.npy`
- `synthetic_data_ctgan.npy`
- `synthetic_data_tvae.npy`
- `info.pkl`
- `average_standardized_log_bids.npy`
- `b_hat.npy`
- `b_tilde_ctgan.npy`
- `b_tilde_tvae.npy`

are in the `multi-level-auction-generator/data/` folder.
**Note that if you have executed the test routine (test_full.sh or test_full.bat) prior to executing the validation routine, the required files mentioned above will have been overwritten by their test versions. As a result, the validation routine will not reproduce the results presented in the paper. The files listed above must either be produced by the full routine (by executing full.sh or full.bat) or be the original files contained in the datasets.zip archive.**

2. Run the validation procedure by executing

- On Mac/Linux, execute:
 `bash validation.sh`

- For Windows users:
 `validation.bat`

This will output the results displayed in Table 2 and 4.

## Run Python Files Individually

Alternatively, python files can be ran individually using, for example,

- On Mac/Linux, execute:
 `python3 src/script/data_transform.py`
 
- For Windows users:
 `python src/script/data_transform.py`

Replace "data_transform.py" by the file you need to run. 

**Note that running training scripts (any Python file ending with "_train.py") with the argument --save_model (e.g., python3 src/script/bidnet_train.py --save_model) will overwrite the associated model parameters stored in multi-level-auction-generator/models/. Running any script may also overwrite data outputs stored in multi-level-auction-generator/data/.** 

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

