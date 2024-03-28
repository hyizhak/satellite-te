#!/bin/bash
source $(dirname $(readlink -f $0))/env

rm -rf $INPUT_DIR
mkdir -p $INPUT_DIR

for problem in $PROBLEM_LIST; do
    cp -r $AZURE_PROBLEM_DIR/$problem $INPUT_DIR
done
