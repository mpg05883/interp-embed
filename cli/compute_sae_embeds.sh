#!/bin/bash
source ./cli/utils.sh
activate_conda_env

dataset="gsm8k"
split="train"
field="answer"
sae_id="blocks.8.hook_resid_pre"
release="gpt2-small-res-jb"

log_info "Starting script: $(basename "$0"). Conda environment: $CONDA_DEFAULT_ENV"
python -m scripts.compute_sae_embeds \
    --dataset ${dataset} \
    --split ${split} \
    --field ${field} \
    --sae_id ${sae_id} \
    --release ${release}