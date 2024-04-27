#!/bin/bash
source $(dirname $(readlink -f $0))/env

INPUT_DIR=/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/raw_data/starlink
OUTPUT_DIR=/home/azureuser/cloudfiles/code/te_problems/starlink

for intensity in 25 50 75 100; do
    nohup python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR}/DataSetForSaTE${intensity} \
        --output-path ${OUTPUT_DIR} \
        --intensity ${intensity} \
        --inter-shell-mode "GrdStation"
done

nohup python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR} \
        --output-path ${OUTPUT_DIR} \
        --inter-shell-mode "GrdStation" \
        --mixed