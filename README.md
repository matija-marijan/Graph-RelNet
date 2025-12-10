# Graph-RelNet
This repository contains the implementation for the paper:

 "Message Passing Neural Networks for Sound Source Localization", Matija Marijan, Miloš Bjelić, presented at the 33rd Telecommunications Forum (TELFOR), Belgrade, Serbia, 2025. 
 
 This repository provides code for training graph neural networks (GNNs) on sound source localization tasks, supporting experiments with various microphone array geometries, synthetic and real audio signals, and deep learning models RelNet and Graph-RelNet. The RelNet model is adapted from [GNN_SSL](https://github.com/egrinstein/gnn_ssl/).

## Directory Structure

- `models/`: Contains model definitions (`Graph-RelNet` and `RelNet`).
- `utils/`: Utility functions for datasets, geometry, signal processing, and general helpers.
- `evaluation/`: Scripts and notebooks for analyzing results.
- `data/`: Default location for generated or real datasets.
- `create_dataset.py`: Script for generating datasets.
- `training.py`: Main training script.
- `prediction.py`: Script for running model inference.
- `run.sh`: Example shell script for running experiments.

## Installation

1. Clone the repository.
2. Create a conda environment using the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate geometric
   ```
3. To download the TIMIT dataset, visit [this link](https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3), unzip the archive, and place its contents under `data/signals/timit/`.

## Usage

- **Dataset Generation**:
  ```bash
  python create_dataset.py --help
  ```
- **Training**:
  ```bash
  python training.py --help
  ```

## Requirements

- Python 3.12
- PyTorch, PyTorch Geometric, and related dependencies (see `environment.yml`)
