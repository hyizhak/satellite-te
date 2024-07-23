import os
import argparse
import sys

ROOT = f'{os.path.dirname(os.path.abspath(__file__))}/../..'
sys.path.append(ROOT)
from lib import AssetManager

parser = argparse.ArgumentParser()
parser.add_argument("--source-problem", type=str, required=True)
parser.add_argument("--target-problems", nargs="+", required=True)
parser.add_argument('--num-path', type=int, default=5)
parser.add_argument('--edge-disjoint', type=bool, default=False)
parser.add_argument('--dist-metric', type=str, default='min-hop', choices=['min-hop', 'inv-cap'])

args = parser.parse_args()

source = args.source_problem
target_list = args.target_problems

num_path = args.num_path
edge_disjoint = args.edge_disjoint
dist_metric = args.dist_metric

topo_num = AssetManager.topo_num(args.source_problem)
for problem in target_list:
    assert(topo_num == AssetManager.topo_num(problem))
    
for problem in target_list:
    print(f'Copying paths from {source} to {problem}')
    for topo_idx in range(topo_num):
        source_file = AssetManager.pathform_path(source, topo_idx, num_path, edge_disjoint, dist_metric)
        target_file = AssetManager.pathform_path(problem, topo_idx, num_path, edge_disjoint, dist_metric, create_path=True)
        os.system(f'cp {source_file} {target_file}')
        