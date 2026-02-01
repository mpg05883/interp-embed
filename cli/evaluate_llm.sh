#!/bin/bash
#SBATCH --job-name=evaluate_llm
#SBATCH --partition=gpuA100x4     
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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
log_info "Starting $(get_slurm_message)"

# Make HF caches go to SCRATCH
CACHE_ROOT="/scratch/bcqc/mgee2/hf_cache"
export HF_HOME="$CACHE_ROOT/hf_home"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

MODEL_NAME="microsoft/Phi-3-mini-128k-instruct"   
TASKS="gsm8k"
BATCH_SIZE=1
OUTDIR="results/evaluation/${TASKS}/${MODEL_NAME//\//_}/${SLURM_JOB_ID}" 
mkdir -p "$OUTDIR"

# TODO: Figure out what args to use to reproduce an LLM's results
if lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_NAME",dtype=float16 \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --output_path "$OUTDIR"; then
    
    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" 
    exit 1
fi