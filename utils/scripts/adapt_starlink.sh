#!/bin/bash
source $(dirname $(readlink -f $0))/env

INPUT_DIR=/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/input/Starlink
OUTPUT_DIR=/home/azureuser/cloudfiles/code/te_problems/starlink

TEST_RATIO=0.2

for intensity in 50; do
    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR}/DataSetForSaTE${intensity} \
        --output-path ${OUTPUT_DIR} \
        --intensity ${intensity}
done