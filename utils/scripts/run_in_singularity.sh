#!/bin/bash

# Define the path to the Singularity image
SINGULARITY_IMAGE="/home/users/nus/e1310988/satte/sing_image/sate_image_latest.sif"

# Define the current directory
CURRENT_DIR=$(pwd)

# Define the paths to mount
source $(dirname $(readlink -f $0))/env

# script to run
SCRIPT_TO_RUN=run_starlink.sh

# Run the Singularity container with mounted directories
singularity exec --nv --bind ${INPUT_DIR}:${INPUT_DIR} --bind ${PROJECT_ROOT}:${PROJECT_ROOT} ${SINGULARITY_IMAGE} bash -c "
  source ${PROJECT_ROOT}/utils/scripts/env && \
  conda init && \
  source ~/.bashrc && \
  conda activate satte && \
  ${SCRIPTS_DIR}/${SCRIPT_TO_RUN}
"