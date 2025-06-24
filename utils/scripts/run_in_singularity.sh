#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <script_to_run>"
  exit 1
fi

# Define the current directory
CURRENT_DIR=$(pwd)

# Define the paths to mount
source $(dirname $(readlink -f $0))/env

# Define the path to the Singularity image
SINGULARITY_IMAGE=${PROJECT_ROOT}/sing_image/sate_image_latest.sif

# script to run
# SCRIPT_TO_RUN=adapt_starlink.sh
SCRIPT_TO_RUN=$1

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
  GPU_FLAG="--nv"
else
  GPU_FLAG=""
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run the Singularity container with mounted directories
singularity exec --cleanenv ${GPU_FLAG} \
  --bind ${INPUT_DIR}:${INPUT_DIR} \
  --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
  ${SINGULARITY_IMAGE} \
  /opt/conda/bin/conda run -n satte --live-stream --no-capture-output \
    "${SCRIPTS_DIR}/${SCRIPT_TO_RUN}" \
  > ${SCRIPTS_DIR}/log/${SCRIPT_TO_RUN}_${PBS_JOBID}_${TIMESTAMP}.log 2>&1