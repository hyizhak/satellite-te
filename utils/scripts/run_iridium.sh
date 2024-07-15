#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

RUN_TOPO_NUM=1

PROBLEM_LIST=$(cd $INPUT_DIR/iridium_new_form; ls)

# PROBLEM_LIST=("IridiumDataSet14day20sec_Int5" "IridiumDataSet14day20sec_Int7p5")

# echo "Problem list: $PROBLEM_LIST"

for problem in Intensity_12p5; do
    echo "Processing problem: $problem"
    # Run Teal
    # python ${TEAL_SCRIPT} \
    #     --problem-path ${INPUT_DIR}/Iridium/${problem} \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}
    # # Run Gurobi
    # python ${LP_SCRIPT} \
    #     --problem-path ${OUTPUT_DIR}/${problem} \
    #     --output-dir ${OUTPUT_DIR} \
    #     --topo-num ${RUN_TOPO_NUM}
    # Run SpaceTE (10 mins per topology/epoch)
    nohup python ${SPACETE_SCRIPT} \
        --problem-path ${INPUT_DIR}/iridium_new_form/${problem} \
        --output-dir ${OUTPUT_DIR} \
        --topo-num ${RUN_TOPO_NUM} \
        --train --test
done

# Copy the output directory to the specified location
cp -r $OUTPUT_DIR ~/cloudfiles/code/Users/e1310988/satellite-te/