#!/bin/bash

source ./cli/utils.sh
source /sw/external/python/anaconda3/etc/profile.d/conda.sh

ENV_NAME="interp-embed-py311"

# Create conda environment if it doesn't exist and activate it
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log_info "Conda environment exists. Activating $ENV_NAME"
else
    log_info "Conda environment does not exist. Creating $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# Install interp_embed packages
pip install git+https://github.com/nickjiang2378/interp_embed
pip install -e .

# Install lm-eval-harness with Hugging Face backend
pip install lm_eval[hf]

# Install Llama CLI
pip install llama-stack


