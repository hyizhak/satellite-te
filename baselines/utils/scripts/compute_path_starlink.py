from itertools import tee, islice
from sys import maxsize
import networkx as nx
import os
import argparse
import multiprocessing as mp
import pickle
import sys
import glob
from pathlib import Path

ARG_PROBLEM_PATH = '~/workspace/input/DataSetForTEAL_100_500'
ARG_OUTPUT_PATH = '~/workspace/pathdict'

ROOT = f'{os.path.dirname(os.path.abspath(__file__))}/../..'
sys.path.append(ROOT)

from lib import AssetManager
from lib.data.starlink import StarlinkPathFormer, InterShellMode as ISM

def _compute_process(problem_path, topo_idx, path_num, ism:ISM, output_prefix):
    
    path_id = StarlinkPathFormer.get_pathdict_id(path_num, ism)
    
    fname = AssetManager.pathdict_path(output_prefix, topo_idx, path_id, True)
    print(f'Computing {fname}...')
    
    metadata = AssetManager.load_pathform_metadata(problem_path, topo_idx)
    pathformer = StarlinkPathFormer(metadata, ism)
    path_dict = pathformer.compute_pathdict(path_num)
    
    with open(fname, 'wb') as f:
        pickle.dump(path_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--num-path', type=int, default=5)
    parser.add_argument('--ism', type=str, default="GrdStation")
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--output-name', type=str)
    parser.add_argument('--parallel', type=int, default=mp.cpu_count())
    
    # get all the subdirectories inside the problem path
    args = parser.parse_args()
    
    problem_path = Path(args.problem_path).expanduser().resolve()
    num_path = args.num_path
    ism = ISM(args.ism)
    output_path = Path(args.output_path).expanduser().resolve()
    output_name = args.output_name if args.output_name != None else problem_path.name
    output_prefix = output_path / output_name
    parallel = args.parallel
    
    print(f'Computing path {output_prefix} for problem: {problem_path}')
    
    topo_num = len(list(problem_path.glob('topo_*')))
    
    params = []
    for i in range(topo_num):
        params.append((problem_path, i, num_path, ism, output_prefix))
    
    with mp.Pool(parallel) as pool:
        pool.starmap(_compute_process, params)
        
    print('Path computation done.')
    