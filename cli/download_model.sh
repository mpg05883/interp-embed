#!/bin/bash
source ./cli/utils.sh
set_env_vars

REPO_ID="meta-llama/Llama-3.3-70B-Instruct"

if python -m scripts.download_model \
    --repo_id ${REPO_ID}; then

    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi

