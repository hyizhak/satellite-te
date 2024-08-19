#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

PROBLEM_LIST=$(cd $INPUT_DIR/starlink; ls)

# PROBLEM_LIST=("IridiumDataSet14day20sec_Int5" "IridiumDataSet14day20sec_Int7p5")

echo "Problem list: $PROBLEM_LIST"

# for mode in ISL GrdStation; do
#     for problem in mixed; do
#         echo -e "\n\n\n Training model on problem: $problem with mode: $mode"

#         python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#             --output-dir ${OUTPUT_DIR}/supervised \
#             --topo-num ${RUN_TOPO_NUM} \
#             --train --test --epochs 10 --lr 0.001 --admm-steps 0 --bsz 4 --supervised
#     done
# done

# for problem in DataSetForSaTE25 DataSetForSaTE50 DataSetForSaTE75 DataSetForSaTE100; do
#     for mode in ISL GrdStation; do
#         echo -e "\n\n\n Testing mix model on problem: $problem with mode: $mode"

#         python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#             --output-dir ${OUTPUT_DIR}/supervised \
#             --topo-num ${RUN_TOPO_NUM} \
#             --test --admm-steps 0 --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/mixed_${mode}_spaceTE/models/spaceTE_obj-teal_total_flow_supervised_lr-0.001_ep-10_sample-200_layers-0_decoder-linear.pt
            
#     done
# done


# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
#     --output-dir ${OUTPUT_DIR}/penalized_optimization \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test --admm-steps 0 --penalized --epochs 200 --flow-lambda 50


# for mode in ISL GrdStation; do
#     echo -e "\n\n\n Training model on problem: starlink_1500 with mode: $mode"

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/starlink_1500/${mode} \
#         --output-dir ${OUTPUT_DIR}/supervised \
#         --topo-num ${RUN_TOPO_NUM} \
#         --train --test --admm-steps 0 --supervised --epochs 10 --lr 0.001 --bsz 4
            
# done


echo -e "\n\n\n Training model on problem: Iridium_15"
python ${SPACETE_SCRIPT} \
    --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
    --output-dir ${OUTPUT_DIR}/supervised \
    --topo-num ${RUN_TOPO_NUM} \
    --train --test --admm-steps 10 --supervised --epochs 10

for problem in starlink_500 starlink_1500 mixed; do
    for mode in ISL GrdStation; do
        echo -e "\n\n\n Training model on problem: $problem with mode: $mode"

        python ${SPACETE_SCRIPT} \
            --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
            --output-dir ${OUTPUT_DIR}/supervised \
            --topo-num ${RUN_TOPO_NUM} \
            --train --test --admm-steps 10 --supervised --epochs 10
            
    done
done

# echo -e "\n\n\n Testing iridium model on problem: Iridium_15"
# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/iridium/new_form/Intensity_15 \
#     --output-dir ${OUTPUT_DIR}/scalability_iridium \
#     --topo-num ${RUN_TOPO_NUM} \
#     --test --admm-steps 10 --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/new_form_Intensity_15_spaceTE/models/spaceTE_obj-teal_total_flow_supervised_lr-0.001_ep-10_sample-200_layers-0_decoder-linear.pt

# for problem in starlink_500 starlink_1500 DataSetForSaTE100; do
#     for mode in ISL GrdStation; do
#         echo -e "\n\n\n Testing iridium model on problem: $problem with mode: $mode"

#         python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#             --output-dir ${OUTPUT_DIR}/scalability_iridium \
#             --topo-num ${RUN_TOPO_NUM} \
#             --test --admm-steps 10 --model-path /data/projects/11003765/sate/satte/satellite-te/output/supervised/new_form_Intensity_15_spaceTE/models/spaceTE_obj-teal_total_flow_supervised_lr-0.001_ep-10_sample-200_layers-0_decoder-linear.pt
            
#     done
# done

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/mixed/ISL \
#     --output-dir ${OUTPUT_DIR}/penalized_optimization \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test --admm-steps 0 --penalized --epochs 10 --bsz 4 --layers 3


# for admm_steps in 0 5 10 15; do
#     echo -e "\n\n\n Testing with admm_steps: $admm_steps"
#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
#         --output-dir ${OUTPUT_DIR}/supervised \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-test --admm-steps ${admm_steps} --supervised \
#         --epochs 5 --lr 0.0001

#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
#         --output-dir ${OUTPUT_DIR}/supervised \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps ${admm_steps} --supervised \
#         --epochs 5 --lr 0.0001
# done

# for problem in DataSetForSaTE100; do
#     for mode in ISL; do
#         for num_layer in 0; do
#             for obj in total_flow rounded_total_flow teal_total_flow; do
#             echo -e "\n\n\n Training model on problem: $problem with mode: $mode and num_layer: $num_layer and obj: $obj"

#             python ${SPACETE_SCRIPT} \
#                 --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#                 --output-dir ${OUTPUT_DIR} \
#                 --topo-num ${RUN_TOPO_NUM} \
#                 --train --epochs 1 --layers ${num_layer} --obj ${obj} --bsz 16 \
#                 --test --admm-steps 0
#             done
#         done
#     done
# done


# for problem in starlink_500 starlink_1500 DataSetForSaTE100; do
#     for mode in ISL GrdStation; do
#         # echo -e "\n\n\n Testing Iridium model on problem: $problem with mode: $mode"
#         echo -e "\n\n\n Training and Testing model on problem: $problem with mode: $mode"

#         # python ${SPACETE_SCRIPT} \
#         #     --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#         #     --output-dir ${OUTPUT_DIR}/scalability \
#         #     --topo-num ${RUN_TOPO_NUM} \
#         #     --test --admm-steps 0 \
#         #     --model-path /data/projects/11003765/sate/satte/satellite-te/output/iridium/new_form_Intensity_15_spaceTE/models/spaceTE_topo-1_tsz-None_vr-0.2_lr-0.0001_ep-1_bsz-32_sample-5_layers-0_rho-1.0_step-10.pt

#         python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#             --output-dir ${OUTPUT_DIR} \
#             --topo-num ${RUN_TOPO_NUM} \
#             --train --test --epochs 1 --admm-steps 0 
#     done
# done