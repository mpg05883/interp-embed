#!/bin/bash

# Returns the current timestamp (Pacific time) formatted as:
# month day year, hour:minute:second AM/PM.
get_timestamp() {
    TZ="America/Los_Angeles" date +"%b %d, %Y %I:%M:%S%p"
}

# Prints timestamped info messages to stdout
log_info() {
    timestamp=$(get_timestamp)
    echo "[${timestamp}] $*"
}

# Prints timestamped error messages to stderr.
log_error() {
    timestamp=$(get_timestamp)
    echo "[${timestamp}] $*" >&2
}


# Activates a hard-coded conda enviornment 
activate_conda_env() {
    source /sw/external/python/anaconda3/etc/profile.d/conda.sh
    conda activate sae-embed
}