#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=2

for problem in $PROBLEM_LIST; do
    # Run Teal
    # python ${TEAL_SCRIPT} \
    #     --problem-path ${INPUT_DIR}/${problem} \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}
    # # Run Gurobi
    # python ${LP_SCRIPT} \
    #     --problem-path ${OUTPUT_DIR}/${problem} \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}
    # Run SpaceTE
    nohup python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/${problem} \
        --output-dir ${OUTPUT_DIR} \
        --topo-num ${RUN_TOPO_NUM} &
done
