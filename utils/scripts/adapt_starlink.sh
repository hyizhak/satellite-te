#!/bin/bash
source $(dirname $(readlink -f $0))/env

RAW_INPUT_DIR=${INPUT_DIR}/raw/starlink

for intensity in 25 50 75 100; do
    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/DataSetForSaTE${intensity} \
        --output-path ${INPUT_DIR}/starlink \
        --intensity ${intensity} \
        --inter-shell-mode "GrdStation"
    
    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/DataSetForSaTE${intensity} \
        --output-path ${INPUT_DIR}/starlink \
        --intensity ${intensity} \
        --inter-shell-mode "ISL"

    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/starlink_500_fixed_topo/Intensity_${intensity} \
        --output-path ${INPUT_DIR}/starlink/starlink_500_fixed_topo \
        --intensity ${intensity} --teal_form \
        --inter-shell-mode "ISL" \
        --reduced 8

    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/starlink_500_fixed_topo/Intensity_${intensity} \
        --output-path ${INPUT_DIR}/starlink/starlink_500_fixed_topo \
        --intensity ${intensity} \
        --inter-shell-mode "ISL" \
        --reduced 8

    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/starlink_500_fixed_topo/Intensity_${intensity} \
        --output-path ${INPUT_DIR}/starlink/starlink_500_fixed_topo \
        --intensity ${intensity} --teal_form \
        --inter-shell-mode "GrdStation" \
        --reduced 8

    python ${STARLINK_ADAPTER_SCRIPT} \
        --input-path ${RAW_INPUT_DIR}/starlink_500_fixed_topo/Intensity_${intensity} \
        --output-path ${INPUT_DIR}/starlink/starlink_500_fixed_topo \
        --intensity ${intensity} \
        --inter-shell-mode "GrdStation" \
        --reduced 8
done

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_500 \
    --output-path ${INPUT_DIR}/starlink \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --reduced 8

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_500 \
    --output-path ${INPUT_DIR}/starlink \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --reduced 8

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_500 \
    --output-path ${INPUT_DIR}/starlink \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --teal_form \
    --reduced 8

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_500 \
    --output-path ${INPUT_DIR}/starlink \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --teal_form \
    --reduced 8

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_1500 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --reduced 2

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_1500 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --reduced 2

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_1500 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --teal_form \
    --reduced 2

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_1500 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --teal_form \
    --reduced 2

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_176 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --reduced 18

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_176 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --reduced 18

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_176 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --teal_form \
    --reduced 18

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_176 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --teal_form \
    --reduced 18

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_528 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "GrdStation" \
    --teal_form \
    --reduced 6

python ${STARLINK_ADAPTER_SCRIPT} \
    --input-path ${RAW_INPUT_DIR}/starlink_528 \
    --output-path ${INPUT_DIR} \
    --intensity 100 \
    --inter-shell-mode "ISL" \
    --teal_form \
    --reduced 6