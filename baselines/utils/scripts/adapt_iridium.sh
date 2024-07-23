#!/bin/bash
set -x

source $(dirname $(readlink -f $0))/env

INPUT_DIR=/home/azureuser/cloudfiles/code/raw_data
OUTPUT_DIR=${PROJECT_ROOT}/input

TEST_RATIO=0.15

for intensity in 10 12p5 15; do
    python ${IRIDIUM_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR}/IridiumDataSet14day20sec_Int${intensity} \
        --output-path ${OUTPUT_DIR} \
        --parallel $(nproc)
done
