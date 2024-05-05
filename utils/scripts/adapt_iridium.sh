#!/bin/bash
source $(dirname $(readlink -f $0))/env

# INPUT_DIR=/home/azureuser/cloudfiles/code/raw_data
# OUTPUT_DIR=/mnt/output

# TEST_RATIO=0.15

# for intensity in 5 7p5 10; do
#     python ${IRIDIUM_ADAPTER_SCRIPT} \
#         --input-path ${INPUT_DIR}/IridiumDataSet14day20sec_Int${intensity} \
#         --output-path ${OUTPUT_DIR}
# done

INPUT_DIR=/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/raw_data/iridium
OUTPUT_DIR=/home/azureuser/cloudfiles/code/te_problems/iridium_new_form

for intensity in 5 7p5 10 12p5 15; do
    nohup python ${IRIDIUM_NEW_FORM_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR} \
        --output-path ${OUTPUT_DIR} \
        --teal-like \
        --intensity ${intensity} &
done