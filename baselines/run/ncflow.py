import os
import pickle
import traceback
from tqdm import tqdm
import sys
import numpy as np
import time
sys.path.append("..")

from lib.lp.problem import Problem

from _common import *
from lib.lp.algorithms import NcfEpi
from lib.lp.constants import NUM_CORES
from lib.lp.partitioning.fm_partitioning import FMPartitioning
from lib.lp.partitioning.leader_election import LeaderElection

from lib.lp.algorithms.load_starlink import construct_from_edge,dict_to_numpy


import argparse

def benchmark(args):
    work_dir = args.work_dir
    scale_factor = args.scale_factor
    test_log_dir = AssetManager.test_log_dir(work_dir, create_dir=True)
    problem_path, topo_num, obj = args.problem_path, args.topo_num, args.obj
    path_num, edge_disjoint, dist_metric = args.path_num, args.edge_disjoint, args.dist_metric
    num_parts_scale_factor = args.num_parts_scale_factor
    if args.test_tm_per_topo is None:
        test_all = True
        test_size_per_topo = None
    else:
        test_all = False
        test_size_per_topo = args.test_tm_per_topo
    output_csv = os.path.join(test_log_dir, f'NCFlow_starlink.csv')
    paths_fname = "/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/input/starlink/DataSetForSaTE25/GrdStation/StarLink_DataSetForAgent25_5000_A.pkl"
    with open(paths_fname, 'rb') as file:
        data = pickle.load(file)
    with open(output_csv, "a") as results:
        print(",".join(TEST_HEADERS), file=results)
        
        for topo_idx in range(len(data)): # For every topology

            data_ = data[topo_idx]
            tm = data_['tm']
            tm = dict_to_numpy(tm,'GRD_STATION')
            # num_paths = data_['path']
            num_paths = 5
            
            G = construct_from_edge(data_['graph'],"GRD_STATION")
            
        
            if test_all:
                test_size_per_topo = 1
            
            for tm_idx in tqdm(range(test_size_per_topo), desc=f"Computing topology {topo_idx}"):
            
                problem = Problem(G, tm, scale_factor=scale_factor)
                problem.name = str(tm_idx)
                traffic_seed = problem.traffic_matrix.seed
                total_demand = problem.total_demand
                ncflow = NcfEpi.new_total_flow(
                        data = data,
                        num_paths = path_num,
                        edge_disjoint=edge_disjoint,
                        dist_metric=dist_metric,
                    )
                partrition = LeaderElection(num_partitions = num_parts_scale_factor)
                ncflow.solve(problem, partrition)
                for iter in range(ncflow.num_iters):
                    nc = ncflow._ncflows[iter]

                    r1_synctime = nc._synctime_dict["r1"]
                    r2_synctime = parallelized_rt(
                            [t for _, t in nc._synctime_dict["r2"].items()], 
                            NUM_CORES)
                    recon_synctime = parallelized_rt(
                            [t for _, t in nc._synctime_dict["reconciliation"].items()], 
                            NUM_CORES)
                    r3_synctime = nc._synctime_dict["r3"]
                    kirchoffs_synctime = parallelized_rt(
                            [t for _, t in nc._synctime_dict["kirchoffs"].items()], 
                            NUM_CORES)

                    (
                        r1_runtime,
                        r2_runtime,
                        recon_runtime,
                        r3_runtime,
                        kirchoffs_runtime,
                    ) = nc.runtime_est(NUM_CORES, breakdown=True)
                    runtime = (
                        r1_runtime
                        + r2_runtime
                        + recon_runtime
                        + r3_runtime
                        + kirchoffs_runtime
                    )
                    total_flow = nc.obj_val

                    result_line = TEST_PLACEHOLDER.format(
                        topo_idx,
                        tm_idx,
                        total_demand,
                        nc.obj_val,
                        nc.obj_val / total_demand,
                        nc.runtime,
                    )
                    print("rate: ", nc.obj_val / total_demand)
                    print(result_line, file=results)


       



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--top-percentage', type=float, default=TOP_PERCENTAGE)
    parser.add_argument("--num-parts-scale-factor",type = int, default=NUM_PARTS_SCALE_FACTOR)
    # problem parameters
    parser.add_argument("--problem-path", type=str, default=ARG_PROBLEM_PATH)
    parser.add_argument('--topo-num', type=int, default=ARG_TOPO_NUM)
    parser.add_argument("--scale-factor", type=float, default=ARG_SCALE_FACTOR, choices=SCALE_FACTORS, help="traffic matrix scale factor")
    parser.add_argument("--dry-run", dest="dry_run", default=False, action="store_true", help="list problems to run")
    parser.add_argument("--obj", type=str, default=ARG_OBJ, choices=OBJ_STRS, help="objective function")

    # output paramenters
    parser.add_argument("--output-dir", type=str, default=ARG_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default=ARG_OUTPUT_PREFIX)
    
    # path form arguments
    parser.add_argument('--path-num', type=int, default=ARG_PATH_NUM)
    parser.add_argument('--edge-disjoint', type=bool, default=ARG_EDGE_DISJOINT)
    parser.add_argument('--dist-metric', type=str, default=ARG_DIST_METIRC)
    
    parser.add_argument('--test-tm-per-topo', type=int, default=ARG_TEST_TM_PER_TOPO)
    
    args = parser.parse_args()
    
    update_output_path(args, "lp")
    print("starting..")
    start_time = time.time()
    try:
        if args.dry_run:
            print("Problem to run: {args.problem_path}\nWorking dir: {args.work_dir}")
        else:
            benchmark(args)
    except Exception as e:
        print(f"程序发生异常: {e}")
    finally:
        # 不管发生什么，finally块中的代码都会执行
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序运行时间：{elapsed_time}秒")
