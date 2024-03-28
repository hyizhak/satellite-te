#! /usr/bin/env python
import argparse
import os
import pickle
from networkx.readwrite import json_graph
import json
from tqdm import tqdm

from _common import *

from lib.lp.algorithms import PathFormulation, Objective
from lib.lp.problem import Problem


# Sweep topos and traffic matrices for that topo. For each combo, record the
# runtime and total flow for each algorithm
def benchmark(args):
    
    problem_path, topo_num, obj = args.problem_path, args.topo_num, args.obj
    scale_factor = args.scale_factor
    path_num, edge_disjoint, dist_metric = args.path_num, args.edge_disjoint, args.dist_metric
    work_dir = args.work_dir
    
    if args.test_tm_per_topo is None:
        test_all = True
        test_size_per_topo = None
    else:
        test_all = False
        test_size_per_topo = args.test_tm_per_topo
    
    test_log_dir = AssetManager.test_log_dir(work_dir, create_dir=True)
    output_csv = os.path.join(test_log_dir, f'gurobi_topo-{topo_num}_tsz-{test_size_per_topo}.csv')
    
    with open(output_csv, "w") as f:
        print(",".join(TEST_HEADERS), file=f)

    logging.info("Gurobi solver")
    logging.info("Problem path: {}, Topology number: {}, Objective: {}".format(problem_path, topo_num, obj))
        
    for topo_idx in range(topo_num): # For every topology

        with open(AssetManager.tm_test_path(problem_path, topo_idx), 'rb') as f:
            tm = pickle.load(f)
            
        with open(AssetManager.graph_path(problem_path, topo_idx), 'r') as f:
            G = json_graph.node_link_graph(json.load(f))       
        
        if test_all:
            test_size_per_topo = tm.shape[0]
            
        for tm_idx in tqdm(range(test_size_per_topo), desc=f"Computing topology {topo_idx}"):
            
            problem = Problem(G, tm[tm_idx], scale_factor=scale_factor)

            pf = PathFormulation(
                problem_path=problem_path,
                topo_idx=topo_idx,
                objective=Objective.get_obj_from_str(obj),
                path_num=path_num,
                edge_disjoint=edge_disjoint,
                dist_metric=dist_metric,
            )
            pf.solve(problem)

            total_demand = problem.total_demand
            
            result_line = TEST_PLACEHOLDER.format(
                topo_idx,
                tm_idx,
                total_demand,
                pf.obj_val,
                pf.obj_val / total_demand,
                pf.runtime,
            )
            
            with open(output_csv, 'a') as f:
                print(result_line, file=f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
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

    if args.dry_run:
        print("Problem to run: {args.problem_path}\nWorking dir: {args.work_dir}")
    else:
        benchmark(args)
