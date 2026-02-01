#!/bin/bash
source ./cli/utils.sh
activate_conda_env

dataset="gsm8k"
split="train"
field="answer"
model="text-embedding-3-large"
batch_size=256

log_info "Starting script: $(basename "$0"). Conda environment: $CONDA_DEFAULT_ENV"
python -m scripts.compute_openai_embeds \
    --dataset ${dataset} \
    --split ${split} \
    --field ${field} \
    --model ${model} \
    --batch_size ${batch_size}