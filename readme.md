We present a deep learning solution to address the challenges of simulating realistic synthetic first-price sealed-bid auction data. The complexities encountered in this type of auction data include high-cardinality discrete feature spaces and a multilevel structure arising from multiple bids associated with a single auction instance. Our methodology combines deep generative modeling (DGM) with an artificial learner that predicts the conditional bid distribution based on auction characteristics, contributing to advancements in simulation-based research. This approach lays the groundwork for creating realistic auction environments suitable for agent-based learning and modeling applications. Our contribution is twofold: we introduce a comprehensive methodology for simulating multilevel discrete auction data, and we underscore the potential of DGM as a powerful instrument for refining simulation techniques and fostering the development of economic models grounded in generative AI.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Download](#data-download)
- [Important](#important)
- [Study Replication](#study-replication)
- [License](#license)

## Requirements 

- Python3.8 or above 
- pip package installer (usually installed automatically with Python)
- 25GB RAM
- GPU access (optional but recommended)
- Mac OS, Linux distribution or Windows

## Installation

(Via command line)

1. Clone the repository

`git clone https://github.com/IgorSadoune/multi-level-auction-generator.git`

2. Virtual environment (optional but recommended)

2.1 Create a virtual environment inside the downloaded repository

Go to the root of the folder "multi-level-auction-generator" and execute 

`python -m venv venv`

2.2 Activate the virtual environment 

Then, the virtual environment needs to be activated when you execute files from this repository. 

- On Mac/Linux, execute:
  `source venv/bin/activate`
  
- On Windows, execute:
  `.\venv\Scripts\activate`

3. Install the required dependencies:

The required python libraries are listed in the "requirements.txt" file. Those can directly be downloaded to your virtual environment (or root system if venv not setup) by executing

`pip install -r requirements.txt`

## Data Download

1. **Download the datasets from** 

- [this link](https://zenodo.org/record/8274020).

2. **Place the downloaded pickle and numpy files (.pkl and .npy)**

- in the ROOT of the "multi-level-auction-generator" folder.

## Study Replication

**Always place yourself at the root of the repository**

There are two ways to replicate the study. 

**Either**

1. **Full Routine** 

Run the entire routine, including the training of the CTGAN, TVAE, BidNet and classifiers for inception scoring. Note that the full routine took approximately 8 hours on a RTX2060.

1.1 Place the pickle file "raw_data.pkl" (only this one) in the folder "data".

1.2 Run the test file which execute the routine with a small amount of data and training iterations

- On Mac/Linux, execute:
 `bash test_full.sh`

- For Windows users:
 `test_full.bat`

1.3 If the test is successful (no error was raised), run the routine using the entirety of the data

- On Mac/Linux, execute:
 `bash full.sh`

- For Windows users:
 `full.bat`

This will train the models and output the results displayed in Table 2, 3, and 4.

**Or**

2. **Validation routine to replicate tables and results**

To replicate the study without training the models again

2.1 Place the files:

- "features_squeezed.npy"
- "synthetic_data_ctgan.npy" 
- "synthetic_data_tvae.npy" 
- "features.npy"
- "standardized_log_average_bids.npy"

in the "multi-level-auction-generator/data/" folder.

2.2 Run the validation procedure by executing

- On Mac/Linux, execute:
 `bash validation.sh`

- For Windows users:
 `validation.bat`

This will output the results displayed in Table 2, 3, and 4.

3. **Run files individually**

Alternatively, python files can be ran individually using for example

- On Mac/Linux, execute:
 `python3 src/script/data_transform.py`
 
- For Windows users:
 `python src/script/data_transform.py`

U+26A0 Note that running a training scripts (any python file ending with "_train.py") with the argument --save_model (e.g., python3 src/script/budnet_train.py --save_model), will erase and replace current model parameters stored in the folder "models". 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

