#!/bin/bash
source $(dirname $(readlink -f $0))/env

cd ${HARP_DIR}

for size in 1500 4000; do

    nohup python3 ${HARP_DIR}/frameworks/gurobi_mlu.py --num_paths_per_pair 5 --opt_start_idx 0 --opt_end_idx 200 --topo starlink_${size} --framework gurobi

    nohup python3 ${HARP_DIR}/run_harp.py --topo starlink_${size} --mode train --epochs 10 --lr 0.007 --batch_size 32 --num_paths_per_pair 5 --num_transformer_layers 3 --num_gnn_layers 6 --num_mlp1_hidden_layers 1 --num_mlp2_hidden_layers 1 --num_for_loops 14  --train_clusters {0..159} --train_start_indices $(printf '0 %.0s' {0..159}) --train_end_indices $(printf '1 %.0s' {0..159}) --val_clusters {160..179} --val_start_indices $(printf '0 %.0s' {160..179}) --val_end_indices $(printf '1 %.0s' {160..179}) --framework harp --pred 0 --dynamic 1 > log/starlink_${size}_harp_train.log 2>&1

    nohup python3 ${HARP_DIR}/run_harp.py --topo starlink_${size} --mode test --num_paths_per_pair 5 --num_for_loops 14  --test_cluster {180..199} --test_start_idx $(printf '0 %.0s' {180..199}) --test_end_idx $(printf '1 %.0s' {180..199}) --framework harp --pred 0 --dynamic 1 > log/starlink_${size}_harp_test.log 2>&1

done

nohup python3 ${HARP_DIR}/frameworks/gurobi_mlu.py --num_paths_per_pair 5 --opt_start_idx 0 --opt_end_idx 200 --topo iridium --framework gurobi

nohup python3 ${HARP_DIR}/run_harp.py --topo iridium --mode train --epochs 10 --lr 0.007 --batch_size 32 --num_paths_per_pair 5 --num_transformer_layers 3 --num_gnn_layers 6 --num_mlp1_hidden_layers 1 --num_mlp2_hidden_layers 1 --num_for_loops 14  --train_clusters {0..159} --train_start_indices $(printf '0 %.0s' {0..159}) --train_end_indices $(printf '1 %.0s' {0..159}) --val_clusters {160..179} --val_start_indices $(printf '0 %.0s' {160..179}) --val_end_indices $(printf '1 %.0s' {160..179}) --framework harp --pred 0 --dynamic 1 > log/iridium_harp_train.log 2>&1

nohup python3 ${HARP_DIR}/run_harp.py --topo iridium --mode test --num_paths_per_pair 5 --num_for_loops 14  --test_cluster {180..199} --test_start_idx $(printf '0 %.0s' {180..199}) --test_end_idx $(printf '1 %.0s' {180..199}) --framework harp --pred 0 --dynamic 1 > log/iridium_harp_test.log 2>&1