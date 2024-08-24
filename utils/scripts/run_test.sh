#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
#     --output-dir ${OUTPUT_DIR}/supervised \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test --admm-steps 10 --supervised --epochs 10

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

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/starlink_1500/ISL \
#     --output-dir ${OUTPUT_DIR}/curriculum_supervised \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test --admm-steps 10 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/curriculum_supervised/starlink_500_ISL_spaceTE/models/spaceTE_supervised-kl_div_ep-10_dummy-path-False_flow-lambda-25_layers-0_base-supervised_new_form_Intensity_15.pt

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
#     --output-dir ${OUTPUT_DIR}/latency \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-steps 10 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/new_form_Intensity_15_spaceTE/models/spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt

# for mode in ISL GrdStation; do

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/starlink_500/${mode} \
#         --output-dir ${OUTPUT_DIR}/latency \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps 10 --supervised --epochs 10 \
#         --model-path /data/projects/11003765/sate/satte/satellite-te/output/curriculum_supervised/starlink_500_${mode}_spaceTE/models/spaceTE_supervised-kl_div_ep-10_dummy-path-False_flow-lambda-25_layers-0_base-supervised_new_form_Intensity_15.pt

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/starlink_1500/${mode} \
#         --output-dir ${OUTPUT_DIR}/latency \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps 10 --supervised --epochs 10 \
#         --model-path /data/projects/11003765/sate/satte/satellite-te/output/curriculum_supervised/starlink_1500_${mode}_spaceTE/models/spaceTE_supervised-kl_div_ep-10_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_starlink_500_${mode}.pt

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/${mode} \
#         --output-dir ${OUTPUT_DIR}/latency \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps 10 --supervised --epochs 10 \
#         --model-path /data/projects/11003765/sate/satte/satellite-te/output/curriculum_supervised/mixed_${mode}_spaceTE/models/spaceTE_supervised-kl_div_ep-10_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_starlink_1500_${mode}.pt

# done

python ${SPACETE_SCRIPT} \
    --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE75/GrdStation \
    --output-dir ${OUTPUT_DIR}/comb_supervised \
    --topo-num ${RUN_TOPO_NUM} \
    --train --test --admm-steps 10 --supervised --penalized --epochs 60 \
    --model-path /data/projects/11003765/sate/satte/satellite-te/output/curriculum_supervised/mixed_GrdStation_spaceTE/models/spaceTE_supervised-kl_div_ep-10_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_starlink_1500_GrdStation.pt

# echo -e "\n\n\n Testing 4k model with 20 step round"

# echo -e "\n\n\n Evenly allocated, 10 admm"
# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/GrdStation \
#     --output-dir ${OUTPUT_DIR}/latency \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-test --admm-steps 10 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/DataSetForSaTE100_GrdStation_spaceTE/models/spaceTE_supervised-kl_div_ep-40_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_DataSetForSaTE100_GrdStation.pt

# echo -e "\n\n\n Evenly allocated, 0 admm"
# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/GrdStation \
#     --output-dir ${OUTPUT_DIR}/latency \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-test --admm-steps 0 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/DataSetForSaTE100_GrdStation_spaceTE/models/spaceTE_supervised-kl_div_ep-40_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_DataSetForSaTE100_GrdStation.pt

# echo -e "\n\n\n Trained model, 10 admm"
# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/GrdStation \
#     --output-dir ${OUTPUT_DIR}/latency \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-steps 10 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/DataSetForSaTE100_GrdStation_spaceTE/models/spaceTE_supervised-kl_div_ep-40_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_DataSetForSaTE100_GrdStation.pt

# echo -e "\n\n\n Trained model, 0 admm"
# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/GrdStation \
#     --output-dir ${OUTPUT_DIR}/latency \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-steps 0 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/comb_supervised/DataSetForSaTE100_GrdStation_spaceTE/models/spaceTE_supervised-kl_div_ep-40_dummy-path-False_flow-lambda-25_layers-0_base-curriculum_supervised_DataSetForSaTE100_GrdStation.pt

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
#     --output-dir ${OUTPUT_DIR}/supervised \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-steps 10 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/new_form_Intensity_15_spaceTE/models/spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
#     --output-dir ${OUTPUT_DIR}/supervised \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-steps 10 --supervised --epochs 10 \
#     --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/mixed_ISL_spaceTE/models/spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt

# for mode in ISL GrdStation; do
#     echo -e "\n\n\n Testing starlink model on problem: Starlink_4000 with mode: $mode"

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/${mode} \
#         --output-dir ${OUTPUT_DIR}/supervised \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps 10 --supervised --epochs 10 --flow-lambda 50 \
#         --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/mix_${mode}_spaceTE/models/spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt
# done