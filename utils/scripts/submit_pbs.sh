#!/bin/bash
# Job name:
#PBS -N run_SaTE
# Output and error files:
#PBS -j oe
#PBS -o run_SaTE.log
# Queue name:
#PBS -q normal
# Resource requests:
#PBS -l select=1:mem=128gb
# Walltime (maximum run time):
#PBS -l walltime=24:30:00
# Project code:
#PBS -P 11003765

# Load Singularity module
module load singularity

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

nvidia-smi

# Run the Singularity run script
./run_in_singularity.sh adapt_starlink.sh