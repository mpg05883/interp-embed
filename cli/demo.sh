#!/bin/bash
source ./cli/utils.sh
activate_conda_env

RELEASE="gemma-scope-9b-pt-res-canonical"
SAE_ID="layer_20/width_131k/canonical"

log_info "Starting script: $(basename "$0"), conda enviornment: $CONDA_DEFAULT_ENV"
if python -m scripts.demo \
    --release ${RELEASE} \
    --sae_id ${SAE_ID}; then
    exit 0
else
    exit 1
fi