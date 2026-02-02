#!/bin/bash
source ./cli/utils.sh
activate_conda_env

release="gemma-scope-9b-pt-res-canonical"
sae_id="layer_20/width_131k/canonical"

log_info "Starting script: $(basename "$0"), conda enviornment: $CONDA_DEFAULT_ENV"
if python -m scripts.demo \
    --release ${release} \
    --sae_id ${sae_id}; then
    exit 0
else
    exit 1
fi