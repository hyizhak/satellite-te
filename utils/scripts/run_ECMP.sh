#!/bin/bash
source $(dirname $(readlink -f $0))/env

# for intensity in 25 50 75 100; do
#     for mode in ISL GrdStation; do
#         nohup python ${ECMP_SCRIPT} ${intensity} ${mode} > ./ECMP_${mode}_${intensity}.out 2>&1 &
#     done
# done

for reduced in 2 8; do
    for mode in ISL GrdStation; do
        nohup python ${ECMP_REDUCED_SCRIPT} ${reduced} ${mode} > ./ECMP_${mode}_${reduced}.out 2>&1 &
    done
done