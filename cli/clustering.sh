#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --partition=gpuA100x4     
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu
#SBATCH --time=4:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

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