#!/bin/python3
import os
import sys
import argparse

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(ROOT)

from lib.data.starlink import IridiumAdapter, InterShellMode as ISM

# ========== Configurations ==========
ARG_TOPO_FILE = 'Iridium_DataSetForAgent_{}_60480.pkl'
ARG_DATA_PER_TOPO = 10000
# ====================================

parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--input-topo-file', type=str, default=ARG_TOPO_FILE)
parser.add_argument('--intensity', type=str, required=True)
parser.add_argument('--data-per-topo', type=int, default=ARG_DATA_PER_TOPO)

parser.add_argument('--prefix', type=str, default=None)
parser.add_argument('--output-path', type=str, required=True)

args = parser.parse_args()

if args.input_path[-1] == '/':
    args.input_path = args.input_path[:-1]

if args.prefix is None:
    args.prefix = f'Intensity_{args.intensity}' 
    
    
output_path = os.path.join(args.output_path, args.prefix)
    
IridiumAdapter(
    input_path=args.input_path,
    topo_file=args.input_topo_file.format(args.intensity),
    data_per_topo=args.data_per_topo,
).adapt(output_path)
