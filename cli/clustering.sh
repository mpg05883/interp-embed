#!/bin/bash
source ./cli/utils.sh
activate_conda_env
set_env_vars

VARIANT_NAME="Llama-3.3-70B-Instruct-SAE-l50"
N_CLUSTERS=8
TOP_N=5

if python -m scripts.clustering \
    --variant_name ${VARIANT_NAME} \
    --n_clusters ${N_CLUSTERS} \
    --top_n ${TOP_N}; then
    
    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exit 1
fi