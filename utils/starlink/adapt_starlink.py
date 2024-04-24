#!/bin/python3
import os
import sys
import argparse

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(ROOT)

from lib.data.starlink import StarlinkAdapter, InterShellMode as ISM

# ========== Configurations ==========
ARG_TOPO_FILE_TEMPLATE = 'StarLink_DataSetForAgent{}_5000_{}.pkl'
FILE_VOLUME = ['A', 'B']
ARG_DATA_PER_TOPO = 5000
ARG_PARALLEL = None
# ====================================

parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--input-topo-file-template', type=str, default=ARG_TOPO_FILE_TEMPLATE)
parser.add_argument('--intensity', type=int, required=True)
parser.add_argument('--data-per-topo', type=int, default=ARG_DATA_PER_TOPO)

parser.add_argument('--inter-shell-mode', type=str, required=True)
parser.add_argument('--parallel', type=int, default=ARG_PARALLEL)

parser.add_argument('--prefix', type=str, default=None)
parser.add_argument('--output-path', type=str, required=True)

args = parser.parse_args()

if args.input_path[-1] == '/':
    args.input_path = args.input_path[:-1]

if args.prefix is None:
    args.prefix = os.path.basename(args.input_path)
    
try:
    args.ism = ISM(args.inter_shell_mode)
except KeyError:
    print(f'Invalid inter-shell mode: {args.inter_shell_mode}')
    exit(1)
    
output_path = os.path.join(args.output_path, args.prefix, args.inter_shell_mode)
    
StarlinkAdapter(
    input_path=args.input_path,
    topo_file_template=args.input_topo_file_template.format(args.intensity, '{}'),
    file_volume=FILE_VOLUME,
    data_per_topo=args.data_per_topo,
    ism=args.ism,
    parallel=args.parallel
).adapt(output_path)
