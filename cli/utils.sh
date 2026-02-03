#!/bin/bash


# Returns the current timestamp (Pacific time) formatted as:
# month day year, hour:minute:second AM/PM.
get_timestamp() {
    TZ="America/Los_Angeles" date +"%b %d, %Y %I:%M:%S%p"
}


# Logs timestamped info message to stdout.
log_info() {
    TIMESTAMP=$(get_timestamp)
    echo "[${TIMESTAMP}] $*"
}


# Logs timestamped error message to stderr.
log_error() {
    TIMESTAMP=$(get_timestamp)
    echo "[${TIMESTAMP}] $*" >&2
}


# Activates a hard-coded conda enviornment.
activate_conda_env() {
    local CONDA_ENV_NAME="interp-embed-py311"
    source /sw/external/python/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
}


# Redirects caches for libaries like Hugging Face, Tranformers, etc. to a 
# hard-coded scratch directory.
redirect_caches() {
    local SCRATCH_DIR="/scratch/bcqc/$USER"

    # Create directory structure
    mkdir -p \
        "$SCRATCH_DIR/hf"/{hub,transformers,datasets,modules} \
        "$SCRATCH_DIR/torch" \
        "$SCRATCH_DIR/triton" \
        "$SCRATCH_DIR/pip-cache" \
        "$SCRATCH_DIR/conda/pkgs" \
        "$SCRATCH_DIR/wandb/cache" \
        "$SCRATCH_DIR/matplotlib" \
        "$SCRATCH_DIR/.cache"

    # Hugging Face
    export HF_DIR="$SCRATCH_DIR/hf"
    export HUGGINGFACE_HUB_CACHE="$HF_DIR/hub"
    export HF_DATASETS_CACHE="$HF_DIR/datasets"
    export HF_MODULES_CACHE="$HF_DIR/modules"

    # PyTorch and Triton
    export TORCH_HOME="$SCRATCH_DIR/torch"
    export TRITON_CACHE_DIR="$SCRATCH_DIR/triton"

    # Python and tooling
    export PIP_CACHE_DIR="$SCRATCH_DIR/pip-cache"
    export CONDA_PKGS_DIRS="$SCRATCH_DIR/conda/pkgs"
    export MPLCONFIGDIR="$SCRATCH_DIR/matplotlib"

    # WANDB
    export WANDB_DIR="$SCRATCH_DIR/wandb"
    export WANDB_CACHE_DIR="$SCRATCH_DIR/wandb/cache"

    # Catch-all cache dir 
    export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
}


# Sets environment variables for things like:
# - Redirecting caches for various libraries to scratch.
set_env_vars() {
    redirect_caches
}

# Returns a "done" directory path for marking job completion and ensures the
# directory exists using.
#
# The directory path depends on whether the job is part of a SLURM array:
# - If part of an array, the path is:
#     ./ouput/logs/<SLURM_JOB_NAME>/done/<SLURM_ARRAY_JOB_ID>
# - Else, the path is:
#     ./ouput/logs/<SLURM_JOB_NAME>/done
#
# - `SLURM_JOB_NAME`: The name of the SLURM job.
# - `SLURM_ARRAY_JOB_ID`: The array job ID, if applicable.
get_done_dir() {
    mkdir -p "./output/logs"
    local BASE_DIR="./output/logs/${SLURM_JOB_NAME}/done"

    if [[ -n "$SLURM_ARRAY_JOB_ID" ]]; then
        DONE_DIR="${BASE_DIR}/${SLURM_ARRAY_JOB_ID}"
    else
        DONE_DIR="$BASE_DIR"
    fi

    mkdir -p "$DONE_DIR"
    echo "$DONE_DIR"
}


# Returns a file path to a "done" file to mark completion of a SLURM job and
# and ensures the file exists.
#
# The file name depends on whether the job is part of a SLURM array:
# - If part of an array, the file is named:
#     <SLURM_ARRAY_TASK_ID>.done
# - Else, the file is named:
#     <SLURM_JOB_ID>.done
#
# The file is placed inside the "done" directory created by `create_done_dir`.
#
# - `SLURM_JOB_ID`: The unique SLURM job ID.
# - `SLURM_ARRAY_JOB_ID`: The ID shared across all array tasks, if applicable.
# - `SLURM_ARRAY_TASK_ID`: The index of the array task, if applicable.
# - `SLURM_JOB_NAME`: The name of the SLURM job (used to determine directory
# path).
get_done_file() {
    local DONE_DIR
    DONE_DIR=$(get_done_dir)

    local DONE_FILE
    if [[ -n "$SLURM_ARRAY_JOB_ID" ]]; then
        DONE_FILE="${SLURM_ARRAY_TASK_ID}.done"
    else
        DONE_FILE="${SLURM_JOB_ID}.done"
    fi

    local DONE_PATH="${DONE_DIR}/${DONE_FILE}"
    touch "$DONE_PATH"
    echo "$DONE_PATH"
}


# Returns a formatted string containing information about the SLURM job like:
# - `SLURM_JOB_NAME`: The name of the SLURM job, or "N/A" if not set.
# - `SLURM_JOB_ID`: The unique SLURM job ID, or "N/A" if not set.
# - `SLURM_ARRAY_JOB_ID`: The array job ID if the job is part of an array, or
#    "N/A" if not set.
# - `SLURM_ARRAY_TASK_ID`: The specific task index within the array job, or 
#   "N/A" if not set.
#
# If the job is part of an array, the returned string includes the array job ID 
# and task ID.
# Otherwise, it only includes the job name and job ID.
# 
# Example output for a job in an array:
#   "Job: train_test, ID: 123456, Array ID: 123456, Task ID: 0"
#
# Example output for a non-array job:
#   "Job: train_test, ID: 123456"
get_slurm_message() {
    NAME="job: ${SLURM_JOB_NAME:-N/A}"
    JOB_ID="ID: ${SLURM_JOB_ID:-N/A}"
    ARRAY_ID="Array ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
    TASK_ID="Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"

    if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
        echo "${NAME}, ${JOB_ID}, ${ARRAY_ID}, ${TASK_ID}"
    else
        echo "${NAME}, ${JOB_ID}"
    fi
}
