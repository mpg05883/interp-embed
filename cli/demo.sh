#!/bin/bash
source ./cli/utils.sh
activate_conda_env
set_env_vars

# Local SAE args
RELEASE="gemma-scope-9b-pt-res-canonical"
SAE_ID="layer_20/width_131k/canonical"

# Goodfire SAE args
VARIANT_NAME="Llama-3.3-70B-Instruct-SAE-l50"

# Specify whether to use local or Goodfire SAE
SAE_TYPE="goodfire"

log_info "Starting script: $(basename "$0"), conda enviornment: $CONDA_DEFAULT_ENV"
if python -m scripts.demo \
    --release ${RELEASE} \
    --sae_id ${SAE_ID} \
    --variant_name ${VARIANT_NAME} \
    --sae_type ${SAE_TYPE}; then
    exit 0
else
    exit 1
fi