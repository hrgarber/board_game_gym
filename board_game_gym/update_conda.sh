#!/bin/bash

# Get the list of all Conda environments
envs=$(conda env list | awk '{print $1}' | grep -v "#" | grep -v "base")

# Define the desired CUDA version
CUDA_VERSION="11.8"  # Replace with your desired CUDA version

# Loop through each environment and install or update CUDA packages
for env in $envs; do
    echo "Updating environment: $env"
    
    # Activate the environment
    conda activate "$env"

    # Check if cudatoolkit is already installed
    if conda list | grep -q 'cudatoolkit'; then
        echo "cudatoolkit already installed in $env. Updating..."
        conda install cudatoolkit=$CUDA_VERSION -n "$env" -y
    else
        echo "Installing cudatoolkit in $env..."
        conda install cudatoolkit=$CUDA_VERSION -n "$env" -y
    fi

    # Check if cudnn is already installed
    if conda list | grep -q 'cudnn'; then
        echo "cudnn already installed in $env. Updating..."
        conda install cudnn -n "$env" -y
    else
        echo "Installing cudnn in $env..."
        conda install cudnn -n "$env" -y
    fi
    
    # Deactivate environment after updating
    conda deactivate
done

echo "All environments have been updated."
