#!/bin/bash

cd ./run
python teal.py --obj total_flow --tm-model toy --constellation Iridium --num-topo 300 --epochs 3 --admm-steps 2 --slice-train-start 0 --slice-train-stop 33600 --slice-val-start 33600 --slice-val-stop 40000 --slice-test-start 40000 --slice-test-stop 50000
