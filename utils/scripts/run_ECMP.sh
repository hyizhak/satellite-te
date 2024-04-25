#!/bin/bash
source $(dirname $(readlink -f $0))/env

for intensity in 25 50 75 100; do
    for mode in ISL GrdStation; do
        nohup python ${ECMP_SCRIPT} ${intensity} ${mode} > ./ECMP_${mode}.out 2>&1 &
    done
done