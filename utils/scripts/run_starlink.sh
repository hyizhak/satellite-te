#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

PROBLEM_LIST=$(cd $INPUT_DIR/starlink; ls)

# PROBLEM_LIST=("IridiumDataSet14day20sec_Int5" "IridiumDataSet14day20sec_Int7p5")

echo "Problem list: $PROBLEM_LIST"

for problem in DataSetForSaTE100; do
    echo "Processing problem: $problem"
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

#     nohup python ${SPACETE_SCRIPT} \
#             --problem-path ${INPUT_DIR}/starlink/${problem}/GrdStation \
#             --output-dir ${OUTPUT_DIR} \
#             --topo-num ${RUN_TOPO_NUM} \
#             --failures 0.05 \
#             --test
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
done

# for problem in starlink_500 starlink_1500; do
#     for mode in ISL GrdStation; do
#         echo -e "\n\n\n Processing problem: $problem with mode: $mode"

#         # python ${SPACETE_SCRIPT} \
#         #     --problem-path ${INPUT_DIR}/${problem}/${mode} \
#         #     --output-dir ${OUTPUT_DIR}/scalability \
#         #     --topo-num ${RUN_TOPO_NUM} \
#         #     --train --test

#         python ${TEAL_SCRIPT} \
#             --problem-path ${INPUT_DIR}/${problem}/${mode}_teal \
#             --output-dir ${OUTPUT_DIR}/teal \
#             --topo-num ${RUN_TOPO_NUM}
#     done
# done

echo -e "\n\n\n Processing problem: Iridium"

python ${TEAL_SCRIPT} \
    --problem-path ${INPUT_DIR}/iridium/IridiumDataSet14day20sec_Int15 \
    --output-dir ${OUTPUT_DIR}/teal \
    --topo-num ${RUN_TOPO_NUM}
