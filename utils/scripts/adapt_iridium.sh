#!/bin/bash
source $(dirname $(readlink -f $0))/env

# for intensity in 5 7p5 10; do
#     python ${IRIDIUM_ADAPTER_SCRIPT} \
#         --input-path ${INPUT_DIR}/IridiumDataSet14day20sec_Int${intensity} \
#         --output-path ${OUTPUT_DIR}
# done

RAW_INPUT_DIR=${INPUT_DIR}/raw
OUTPUT_DIR=${INPUT_DIR}/iridium/new_form

for intensity in 15; do
    python ${IRIDIUM_NEW_FORM_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR} \
        --output-path ${OUTPUT_DIR} \
        --intensity ${intensity} 
done