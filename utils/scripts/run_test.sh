#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
#     --output-dir ${OUTPUT_DIR}/mix_supervised \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test --admm-steps 10 --supervised --penalized --epochs 10 --flow-lambda 50

# for mode in ISL GrdStation; do
#     for problem in starlink_500 starlink_1500 DataSetForSaTE100; do
#         echo -e "\n\n\n Testing iridium model on problem: $problem with mode: $mode"

#         python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#             --output-dir ${OUTPUT_DIR}/scalability_iridium \
#             --topo-num ${RUN_TOPO_NUM} \
#             --test --admm-steps 10 --supervised --epochs 10 --flow-lambda 50 \
#             --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/new_form_Intensity_15_spaceTE/models/spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt
#     done
# done

for mode in ISL GrdStation; do
    for problem in starlink_500 starlink_1500 mixed; do
        echo -e "\n\n\n Training and Testing model on problem: $problem with mode: $mode"

        python ${SPACETE_SCRIPT} \
            --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
            --output-dir ${OUTPUT_DIR}/comb_supervised \
            --topo-num ${RUN_TOPO_NUM} \
            --train --test --admm-steps 10 --supervised --penalized --epochs 20 --flow-lambda 50 
    done
done

# for mode in ISL GrdStation; do
#     echo -e "\n\n\n Testing starlink model on problem: Starlink_4000 with mode: $mode"

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/${mode} \
#         --output-dir ${OUTPUT_DIR}/supervised \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps 10 --supervised --epochs 10 --flow-lambda 50 \
#         --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/mix_${mode}_spaceTE/models/spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt
# done