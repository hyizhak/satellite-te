#!/bin/bash
source $(dirname $(readlink -f $0))/env

INPUT_DIR=/home/azureuser/cloudfiles/code/raw_data
OUTPUT_DIR=/mnt/output

TEST_RATIO=0.15

for intensity in 5 7p5 10; do
    python ${IRIDIUM_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR}/IridiumDataSet14day20sec_Int${intensity} \
        --output-path ${OUTPUT_DIR}
done
