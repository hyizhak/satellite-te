import os
import pickle
import traceback
from tqdm import tqdm
import sys
import time

sys.path.append("..")

from lib.lp.problem import Problem
from lib.lp.algorithms import TopFormulation, PathFormulation, Objective
from lib.lp.traffic_matrix import *
from lib.lp.algorithms.load_starlink import construct_from_edge,dict_to_numpy, pop_split

from _common import *

import argparse

INTENSITY = 100
MODE = 'ISL'
ONLINE = False
SIZE = 5000

# Sweep topos and traffic matrices for that topo. For each combo, record the
# runtime and total flow for each algorithm
def benchmark(args):
    top_percentage = args.top_percentage
    num_paths, edge_disjoint, dist_metric = args.path_num,False, args.dist_metric
    work_dir = args.work_dir
    scale_factor = args.scale_factor
    intensity = args.intensity
    size = args.size
    mode = args.mode
    online = args.online
    POP_ratio = args.POP_ratio

    problem_path, topo_num, obj = args.problem_path, args.topo_num, args.obj
    
    logging.info("LPTop solver")
    if size == 5000:
        paths_fname = f"/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/input/starlink/DataSetForSaTE{intensity}/{mode}/StarLink_DataSetForAgent{intensity}_10_A.pkl"
    else:
        paths_fname = f'/home/azureuser/cloudfiles/code/te_problems/dataset_sample/{mode}/StarLink_DataSetForAgent100_10_Size{size}.pkl'
    with open(paths_fname, 'rb') as file:
        data = pickle.load(file)
    logging.info(f"Loaded data from {paths_fname}")

    for topo_idx in range(1): # For every topology

        if POP_ratio != 1:
            data_ = data[topo_idx]
            tm = data_['tm']
            tm = dict_to_numpy(tm, mode, size)
            # num_paths = data_['path']
            num_paths = 5
            G = construct_from_edge(data_['graph'], mode, online, 1, size)
            start_time = time.time()
            logging.info('seperating tms')
            tms = pop_split(tm, POP_ratio)
            logging.info('seperated tms')
            split_time = time.time() - start_time

            demand = 0
            throughput = 0

            start_time = time.time()

            problem = Problem(G, scale_factor=scale_factor)
            # pf = TopFormulation(
            #         data = data_,
            #         top_percentage=top_percentage,
            #         objective=Objective.get_obj_from_str(obj),
            #         path_num=num_paths,
            #         edge_disjoint=edge_disjoint,
            #         dist_metric=dist_metric,
            #         mode=mode
            #     )
            
            pf = PathFormulation(
                data=data_,
                objective=Objective.get_obj_from_str(obj),
                path_num=num_paths,
                edge_disjoint=edge_disjoint,
                dist_metric=dist_metric,
            )

            for i in range(POP_ratio):
                problem.traffic_matrix = GenericTrafficMatrix(problem, tms[i])
                pf.solve(problem)
                total_demand = problem.total_demand
                demand += total_demand
                throughput += pf.obj_val
            runtime = time.time() - start_time
            # logging.info(pf.buildtime)
            logging.info(pf.runtime)
            result_line = TEST_PLACEHOLDER.format(
                topo_idx,
                demand,
                throughput,
                throughput / demand,
                split_time + runtime,
            )
            logging.info(split_time)
            logging.info(runtime)
            with open(args.output_csv, 'a') as f:
                print(result_line, file=f)
    
        else:

            data_ = data[topo_idx]
            tm = data_['tm']
            tm = dict_to_numpy(tm, mode, size)
            # num_paths = data_['path']
            num_paths = 5
                
            G = construct_from_edge(data_['graph'], mode, online, POP_ratio, size)

            start_time = time.time()
            #logging.info(f"{tm.shape[0]},{len(G)}")
            problem = Problem(G, tm, scale_factor=scale_factor)
            if top_percentage != 1:
                pf = TopFormulation(
                    data = data_,
                    top_percentage=top_percentage,
                    objective=Objective.get_obj_from_str(obj),
                    path_num=num_paths,
                    edge_disjoint=edge_disjoint,
                    dist_metric=dist_metric,
                    mode=mode
                )
            else:
                pf = PathFormulation(
                    data=data_,
                    objective=Objective.get_obj_from_str(obj),
                    path_num=num_paths,
                    edge_disjoint=edge_disjoint,
                    dist_metric=dist_metric,
                )
            pf.solve(problem)

            total_demand = problem.total_demand
            result_line = TEST_PLACEHOLDER.format(
                topo_idx,
                total_demand,
                pf.obj_val,
                pf.obj_val / total_demand,
                time.time() - start_time,
            )
            with open(args.output_csv, 'a') as f:
                print(result_line, file=f)

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--top-percentage', type=float, default=TOP_PERCENTAGE)
    # problem parameters
    parser.add_argument("--problem-path", type=str, default=ARG_PROBLEM_PATH)
    parser.add_argument('--topo-num', type=int, default=ARG_TOPO_NUM)
    parser.add_argument("--scale-factor", type=float, default=ARG_SCALE_FACTOR, choices=SCALE_FACTORS, help="traffic matrix scale factor")
    parser.add_argument("--dry-run", dest="dry_run", default=False, action="store_true", help="list problems to run")
    parser.add_argument("--obj", type=str, default=ARG_OBJ, choices=OBJ_STRS, help="objective function")
    parser.add_argument("--intensity", type=int, default=INTENSITY)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--online", type=bool, default=ONLINE)
    parser.add_argument("--POP-ratio", type=int, default=1)

    # output paramenters
    parser.add_argument("--output-dir", type=str, default=ARG_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str)
    
    # path form arguments
    parser.add_argument('--path-num', type=int, default=ARG_PATH_NUM)
    parser.add_argument('--edge-disjoint', type=bool, default=ARG_EDGE_DISJOINT)
    parser.add_argument('--dist-metric', type=str, default=ARG_DIST_METIRC)
    
    parser.add_argument('--test-tm-per-topo', type=int, default=ARG_TEST_TM_PER_TOPO)
    
    args = parser.parse_args()
    
    update_output_path(args, f"lp_TOP-{args.top_percentage}_{args.intensity}_{args.size}_{args.mode}_Online-{args.online}_POP-{args.POP_ratio}")
    

    if args.dry_run:
        print("Problem to run: {args.problem_path}\nWorking dir: {args.work_dir}")
    else:
        benchmark(args)
