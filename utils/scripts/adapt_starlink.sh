#!/bin/bash
source $(dirname $(readlink -f $0))/env

INPUT_DIR=/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/raw_data
OUTPUT_DIR=/home/azureuser/cloudfiles/code/te_problems/starlink

# for intensity in 25 50 75 100; do
#     nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${INPUT_DIR}/starlink/DataSetForSaTE${intensity} \
#         --output-path ${OUTPUT_DIR} \
#         --intensity ${intensity} \
#         --inter-shell-mode "GrdStation"
    
#     nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${INPUT_DIR}/starlink/DataSetForSaTE${intensity} \
#         --output-path ${OUTPUT_DIR} \
#         --intensity ${intensity} \
#         --inter-shell-mode "ISL"
# done

# python${STARLINK_ADAPTER_SCRIPT} \
#     --input-path ${INPUT_DIR}/starlink_500 \
#     --output-path ${OUTPUT_DIR} \
#     --intensity 100 \
#     --inter-shell-mode "GrdStation" \
#     --reduced 8 

# python ${STARLINK_ADAPTER_SCRIPT} \
#     --input-path ${INPUT_DIR}/starlink_500 \
#     --output-path ${OUTPUT_DIR} \
#     --intensity 100 \
#     --inter-shell-mode "ISL" \
#     --reduced 8 

python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR}/starlink_1500 \
        --output-path ${OUTPUT_DIR} \
        --intensity 100 \
        --inter-shell-mode "GrdStation" \
        --data-per-topo 100 \
        --teal_form \
        --reduced 2

python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${INPUT_DIR}/starlink_1500 \
        --output-path ${OUTPUT_DIR} \
        --intensity 100 \
        --inter-shell-mode "ISL" \
        --data-per-topo 100 \
        --teal_form \
        --reduced 2

# nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${INPUT_DIR}/starlink \
#         --output-path ${OUTPUT_DIR} \
#         --inter-shell-mode "GrdStation" \
#         --mixed

# nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${INPUT_DIR}/starlink \
#         --output-path ${OUTPUT_DIR} \
#         --inter-shell-mode "ISL" \
#         --mixed