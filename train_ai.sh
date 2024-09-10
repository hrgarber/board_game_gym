#!/bin/bash

# Activate the conda environment
conda activate pathz_env

# Install required packages
pip install -r requirements.txt

# Run the training script with any provided arguments
python src/backend/train_ai.py "$@"

# Deactivate the conda environment
conda deactivate