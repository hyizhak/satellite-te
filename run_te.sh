#!/bin/bash

cd ./run
# python teal.py --obj total_flow --tm-model toy --constellation Iridium --num-topo 10 --epochs 3 --admm-steps 2 --slice-train-start 0 --slice-train-stop 30 --slice-val-start 30 --slice-val-stop 40 --slice-test-start 40 --slice-test-stop 50
# python teal.py --obj total_flow --tm-model toy --constellation Iridium --num-topo 300 --epochs 3 --admm-steps 2 --slice-train-start 0 --slice-train-stop 3360 --slice-val-start 3360 --slice-val-stop 4000 --slice-test-start 4000 --slice-test-stop 5000 --model-save True
python teal.py --obj total_flow --tm-model toy --constellation Iridium --num-topo 1 --epochs 10 --admm-steps 2 --slice-train-start 36 --slice-train-stop 148 --slice-val-start 148 --slice-val-stop 176 --slice-test-start 176 --slice-test-stop 196

