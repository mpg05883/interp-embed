#!/bin/bash
source ./cli/utils.sh
activate_conda_env
log_info "Starting script: $(basename "$0"). Conda enviornment: $CONDA_DEFAULT_ENV"
python -m scripts.demo