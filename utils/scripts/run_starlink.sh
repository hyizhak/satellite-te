#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

PROBLEM_LIST=$(cd $INPUT_DIR/starlink; ls)

# PROBLEM_LIST=("IridiumDataSet14day20sec_Int5" "IridiumDataSet14day20sec_Int7p5")

echo "Problem list: $PROBLEM_LIST"

# for problem in DataSetForSaTE100; do
    # echo "Processing problem: $problem"
    # Run Teal
    # python ${TEAL_SCRIPT} \
    #     --problem-path ${INPUT_DIR}/starlink/${problem}/ISL_teal \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}

    # python ${TEAL_SCRIPT} \
    #     --problem-path ${INPUT_DIR}/starlink/${problem}/GrdStation_teal \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}


    # # Run Gurobi
    # python ${LP_SCRIPT} \
    #     --problem-path ${OUTPUT_DIR}/${problem} \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}


    # Run SpaceTE (need to change the params in spaceTE.py)
    # nohup python ${SPACETE_SCRIPT} \
    #         --problem-path ${INPUT_DIR}/starlink/${problem}/ISL \
    #         --output-dir ${OUTPUT_DIR} \
    #         --topo-num ${RUN_TOPO_NUM} \
    #         --failures 0.001 \
    #         --test

    # python ${SPACETE_SCRIPT} \
    #         --problem-path ${INPUT_DIR}/starlink/${problem}/ISL \
    #         --output-dir ${OUTPUT_DIR} \
    #         --topo-num ${RUN_TOPO_NUM} \
    #         --failures 0.05 \
    #         --test

    # nohup python ${SPACETE_SCRIPT} \
    #         --problem-path ${INPUT_DIR}/starlink/${problem}/GrdStation \
    #         --output-dir ${OUTPUT_DIR} \
    #         --topo-num ${RUN_TOPO_NUM} \
    #         --failures 0.001 \
    #         --train --test

    # nohup python ${SPACETE_SCRIPT} \
    #         --problem-path ${INPUT_DIR}/starlink/${problem}/GrdStation \
    #         --output-dir ${OUTPUT_DIR} \
    #         --topo-num ${RUN_TOPO_NUM} \
    #         --failures 0.05 \
    #         --test
    # nohup python ${SPACETE_SCRIPT} \
    #         --problem-path ${INPUT_DIR}/starlink/${problem}/GrdStation \
    #         --output-dir ${OUTPUT_DIR} \
    #         --topo-num ${RUN_TOPO_NUM} \
    #         --train --test

    # nohup python ${SPACETE_SCRIPT} \
    #         --problem-path ${INPUT_DIR}/starlink/${problem}/ISL \
    #         --output-dir ${OUTPUT_DIR} \
    #         --topo-num ${RUN_TOPO_NUM} \
    #         --train --test
# done

# echo -e "\n\n\n Processing problem: Iridium"

# python ${TEAL_SCRIPT} \
#     --problem-path ${INPUT_DIR}/iridium/IridiumDataSet14day20sec_Int15 \
#     --output-dir ${OUTPUT_DIR}/teal \
#     --topo-num ${RUN_TOPO_NUM}

# for problem in starlink_176; do
#     for mode in ISL GrdStation; do
#         echo -e "\n\n\n Processing problem: $problem with mode: $mode"

#         python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/${mode} \
#             --output-dir ${OUTPUT_DIR}/scalability \
#             --topo-num ${RUN_TOPO_NUM} \
#             --train --test

#         # python ${TEAL_SCRIPT} \
#         #     --problem-path ${INPUT_DIR}/starlink/${problem}/${mode}_teal \
#         #     --output-dir ${OUTPUT_DIR}/teal \
#         #     --topo-num ${RUN_TOPO_NUM}
#     done
# done

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/starlink_1500/GrdStation \
#     --output-dir ${OUTPUT_DIR}/scalability \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test

# python ${SPACETE_SCRIPT} \
#     --problem-path ${INPUT_DIR}/starlink/starlink_1500/ISL \
#     --output-dir ${OUTPUT_DIR}/scalability \
#     --topo-num ${RUN_TOPO_NUM} \
#     --train --test

python ${SPACETE_SCRIPT} \
    --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
    --output-dir ${OUTPUT_DIR}/supervised \
    --topo-num ${RUN_TOPO_NUM} \
    --train --test --supervised

# for admm_steps in 0 5 10 15; do
#     echo -e "\n\n\n Testing with admm_steps: $admm_steps"
#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
#         --output-dir ${OUTPUT_DIR} \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-test --admm-steps ${admm_steps} --obj total_flow
    
#     python ${SPACETE_SCRIPT} \
#         --problem-path ${INPUT_DIR}/starlink/DataSetForSaTE100/ISL \
#         --output-dir ${OUTPUT_DIR} \
#         --topo-num ${RUN_TOPO_NUM} \
#         --test --admm-steps ${admm_steps} --obj total_flow
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