# SaTE: Low-Latency Traffic Engineering for Satellite Networks

This repository accompanies our paper **"SaTE: Low-Latency Traffic Engineering for Satellite Networks"**. It provides source codes, along with the necessary scripts, tools, and environment to reproduce the results discussed in the paper. 

## Overview

The repository includes:
- Source codes for SaTE the GNN system we proposed
- A pre-configured Singularity/Dokcer image that encapsulates all dependencies and tools needed for running the experiments.
- A collection of bash scripts located in `satellite-te/utils/scripts/` that allow you to replicate the results presented in the paper.

## Prerequisites

### 1. Hardware requirements

We conducted our experiments on a Standard_NC24ads_A100_v4 instance on Microsoft Azure and cluster at National Supercomputing Center Singapore (we requested 110gb memory and 1 A100 GPU for experiments).

### 2. Environment

To set up the environment, pull the image containing all necessary software dependencies:

```bash
singularity pull docker://hyizhak/sate_image_ot
```
Or create (and activate) the conda virtual environment with the environment.yaml we provide:

```bash
conda env create -f environment.yaml
conda activate satte
```

### 3. Runing Experiments

To run the experiments, please follow the steps below:

- Create a directory (e.g., `.../raw/starlink`).
- Download the Starlink input dataset from: https://drive.google.com/drive/folders/1h6kbOj4HpqofPNd7lkIJDTut4XF4ipAF?usp=sharing. Save the downloaded datasets (DataSetForSaTE25, DataSetForSaTE50, DataSetForSaTE75, and DataSetForSaTE100) into this directory.
- Update the variables INPUT_DIR and RAW_INPUT_DIR in utils/scripts/env to point to the directory containing the downloaded data.
- Run the corresponding adaptation scripts (e.g., utils/scripts/adapt_starlink.sh) to preprocess the dataset, including path pre-configuration and traffic aggregation.


Each experiment presented in the paper can be reproduced using the provided bash scripts. This repository consists mainly of:

```plaintext
satellite-te/
├── analyze/                        # Quick validation and result processing
├── baselines/                      # Baselines using Gurobi including LP, POP, TOP 
├── lib/                            # Source code of SaTE and another learning-based baseline Teal
├── output/                         # Trained models and testing reports
├── run/                            # Main classes scripts call
├── utils/
│   └── scripts/
│       ├── run_GS.sh               # Script for experiments on satellites linked with ground relays
│       ├── run_ISL.sh              # Script for experiments on satellites linked with laser links
│       ├── run_in_singularity.sh   # Script to facilitate experiments with singularity images
│       ├── log/                    # Logs from tasks on NSCC clusters             
│       └── ...
├── ...
└── README.md                       # This file
```

To execute an experiment, navigate to the directory and run the corresponding part in the bash file. For example to run experiments with ISL inter-satellite linkage mode:

```bash
cd satellite-te/utils/scripts
bash run_ISL.sh
```

We recommend to run with the container, for example:

```bash
bash run_in_singularity.sh run_ISL.sh
```

