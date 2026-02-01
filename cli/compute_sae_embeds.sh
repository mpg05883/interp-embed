#!/bin/bash
#SBATCH --job-name=compute_sae_embeds
#SBATCH --partition=gpuA100x4     
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu
#SBATCH --time=2:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ./cli/utils.sh
activate_conda_env

dataset="gsm8k"
split="train"
field="answer"
sae_id="blocks.8.hook_resid_pre"
release="gpt2-small-res-jb"

log_info "Starting $(get_slurm_message), script: $(basename "$0"). Conda environment: $CONDA_DEFAULT_ENV"
if python -m scripts.compute_sae_embeds \
    --dataset ${dataset} \
    --split ${split} \
    --field ${field} \
    --sae_id ${sae_id} \
    --release ${release}; then 
    
    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi