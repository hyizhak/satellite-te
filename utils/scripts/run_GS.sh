#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

PROBLEM_LIST=$(cd $INPUT_DIR/starlink; ls)

echo "Problem list: $PROBLEM_LIST"

mode=GrdStation

# Fig 9 a
problem=DataSetForSaTE100

for pruning in 4 8 16 32 64 128 256 512; do
    echo -e "\n\n\n Testing pruning with $pruning on problem: $problem with mode: $mode"

    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
        --output-dir ${OUTPUT_DIR}/isomorphism_pruning \
        --topo-num ${RUN_TOPO_NUM} --pruning-to ${pruning} \
        --train --test --admm-steps 0 --supervised --penalized --epochs 60 
done

# Fig 10 a b
for problem in DataSetForSaTE25 DataSetForSaTE50 DataSetForSaTE75 DataSetForSaTE100; do
    echo -e "\n\n\n Training and Testing SaTE model on problem: $problem with mode: $mode"

    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
        --output-dir ${OUTPUT_DIR}/comb_supervised \
        --topo-num ${RUN_TOPO_NUM} \
        --train --test --admm-steps 0 --supervised --penalized --epochs 60 
done

# Fig 10 c
for intensity in 25 50 75 100; do
    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/starlink_500_fixed_topo/Intensity_${intensity}/${mode} \
        --output-dir ${OUTPUT_DIR}/comb_supervised \
        --topo-num ${RUN_TOPO_NUM} \
        --train --test --admm-steps 0 --supervised --penalized --epochs 12 

    python ${TEAL_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/starlink_500_fixed_topo/Intensity_${intensity}/${mode}_teal \
        --output-dir ${OUTPUT_DIR}/teal \
        --topo-num ${RUN_TOPO_NUM} --obj total_flow
done

# Fig 10 d
for problem in DataSetForSaTE25 DataSetForSaTE50 DataSetForSaTE75 DataSetForSaTE100; do
    echo -e "\n\n\n Training and Testing MLU model on problem: $problem with mode: $mode"

    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
        --output-dir ${OUTPUT_DIR}/comb_supervised \
        --topo-num ${RUN_TOPO_NUM} \
        --train --test --admm-steps 0 --obj teal_min_max_link_util --epochs 8 
done

# Fig 11 a
python ${SPACETE_SCRIPT} \
    --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
    --output-dir ${OUTPUT_DIR}/scalability_500 \
    --topo-num ${RUN_TOPO_NUM} \
    --train --test --admm-steps 0 --supervised --penalized --epochs 12

for problem in starlink500 starlink1500; do
    echo -e "\n\n\n Training and Testing SaTE model on problem: $problem with mode: $mode"

    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
        --output-dir ${OUTPUT_DIR}/comb_supervised \
        --topo-num ${RUN_TOPO_NUM} \
        --train --test --admm-steps 0 --supervised --penalized --epochs 12 
done

echo -e "\n\n\n Testing 500 model on problem: Iridium_15"
python ${SPACETE_SCRIPT} \
    --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
    --output-dir ${OUTPUT_DIR}/scalability_500 \
    --topo-num ${RUN_TOPO_NUM} \
    --test --admm-steps 0 --model-path ${OUTPUT_DIR}/comb_supervised/starlink_500_ISL_spaceTE/models/spaceTE_supervised-kl_div_ep-12_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_starlink_500_ISL/epoch_12.pt

for problem in starlink_1500 DataSetForSaTE100; do
    echo -e "\n\n\n Testing 500 model on problem: $problem with mode: $mode"

    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
        --output-dir ${OUTPUT_DIR}/scalability_500 \
        --topo-num ${RUN_TOPO_NUM} \
        --test --admm-steps 0 --model-path ${OUTPUT_DIR}/comb_supervised/starlink_500_ISL_spaceTE/models/spaceTE_supervised-kl_div_ep-12_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_starlink_500_ISL/epoch_12.pt 
done

# Fig 11 b
for failure in 0 0.001 0.01 0.05; do
    echo -e "\n\n\n Testing model on problem: $problem with mode: $mode with failures: $failure"

    python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/${mode} \
        --output-dir ${OUTPUT_DIR}/comb_supervised \
        --topo-num ${RUN_TOPO_NUM} \
        --test --admm-steps 0 --supervised --failures $failure\
        --model-path ${OUTPUT_DIR}/comb_supervised/DataSetForSaTE100_${mode}_spaceTE/models/spaceTE_supervised-kl_div_ep-60_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_mixed_${mode}/epoch_60.pt
done