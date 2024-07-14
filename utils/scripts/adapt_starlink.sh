#!/bin/bash
source $(dirname $(readlink -f $0))/env

RAW_INPUT_DIR=${INPUT_DIR}/raw
# for intensity in 25 50 75 100; do
#     nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${RAW_INPUT_DIR}/starlink/DataSetForSaTE${intensity} \
#         --output-path ${INPUT_DIR} \
#         --intensity ${intensity} \
#         --inter-shell-mode "GrdStation"
    
#     nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${RAW_INPUT_DIR}/starlink/DataSetForSaTE${intensity} \
#         --output-path ${INPUT_DIR} \
#         --intensity ${intensity} \
#         --inter-shell-mode "ISL"
# done

# python${STARLINK_ADAPTER_SCRIPT} \
#     --input-path ${RAW_INPUT_DIR}/starlink_500 \
#     --output-path ${INPUT_DIR} \
#     --intensity 100 \
#     --inter-shell-mode "GrdStation" \
#     --reduced 8 

# python ${STARLINK_ADAPTER_SCRIPT} \
#     --input-path ${RAW_INPUT_DIR}/starlink_500 \
#     --output-path ${INPUT_DIR} \
#     --intensity 100 \
#     --inter-shell-mode "ISL" \
#     --reduced 8 

python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/starlink_500 \
        --output-path ${INPUT_DIR} \
        --intensity 100 \
        --inter-shell-mode "GrdStation" \
        --data-per-topo 100 \
        --teal_form \
        --reduced 8

python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/starlink_500 \
        --output-path ${INPUT_DIR} \
        --intensity 100 \
        --inter-shell-mode "ISL" \
        --data-per-topo 100 \
        --teal_form \
        --reduced 8

# nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${RAW_INPUT_DIR}/starlink \
#         --output-path ${INPUT_DIR} \
#         --inter-shell-mode "GrdStation" \
#         --mixed

# nohup python ${STARLINK_ADAPTER_SCRIPT} \
#         --input-path ${RAW_INPUT_DIR}/starlink \
#         --output-path ${INPUT_DIR} \
#         --inter-shell-mode "ISL" \
#         --mixed