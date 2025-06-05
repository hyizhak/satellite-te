#!/bin/bash
source $(dirname $(readlink -f $0))/env

RAW_INPUT_DIR=${INPUT_DIR}/raw/starlink

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/DataSetForSaTE100 \
    --output-path ${HARP_DIR} \
    --intensity 100 \
    --harp_form \
    --inter-shell-mode "ISL"

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_500 \
    --output-path ${HARP_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --harp_form \
    --reduced 8

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_1500 \
    --output-path ${HARP_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --harp_form \
    --reduced 2

RAW_INPUT_DIR=${INPUT_DIR}/raw

for intensity in 15; do
    python ${IRIDIUM_NEW_FORM_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR} \
        --output-path ${HARP_DIR} \
        --harp_form \
        --intensity ${intensity} 
done

