#!/bin/bash
set -xtrace

source $(dirname $(readlink -f $0))/env

python3 ${UTILS_DIR}/pathform/copy_path.py --source-problem ../../input/IridiumDataSet14day20sec_Int5 \
    --target-problems ${AZURE_PROBLEM_DIR}/IridiumDataSet14day20sec_Int5 \
    ${AZURE_PROBLEM_DIR}/IridiumDataSet14day20sec_Int7p5 \
    ${AZURE_PROBLEM_DIR}/IridiumDataSet14day20sec_Int10
     