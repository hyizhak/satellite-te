import os
import pickle
import traceback
from tqdm import tqdm
import sys

sys.path.append("..")

from lib.lp.problem import Problem
from lib.lp.algorithms import POP, Objective, PathFormulation, TEAVAR

from lib.lp.benchmark_helpers import AlgoClsAction


from _common import *

from lib.lp.algorithms.load_starlink import construct_from_edge,dict_to_numpy
import argparse


def benchmark(args):
    obj = args.obj
    algo_cls = PathFormulation
    num_subproblems = args.num_subproblems
    split_method = args.split_methods
    split_fraction = args.split_fractions
    problem_path, topo_num, obj = args.problem_path, args.topo_num, args.obj
    scale_factor = args.scale_factor
    path_num, edge_disjoint, dist_metric = args.path_num, args.edge_disjoint, args.dist_metric
    work_dir = args.work_dir
    
    
    if args.test_tm_per_topo is None:
        test_all = True
        test_size_per_topo = None
    else:
        test_all = False
        test_size_per_topo = 1
    
    test_log_dir = AssetManager.test_log_dir(work_dir, create_dir=True)
    output_csv = os.path.join(test_log_dir, f'pop_topo-{topo_num}_tsz-{test_size_per_topo}.csv')
    num_paths, edge_disjoint, dist_metric = (5, False, "min-hop")
    paths_fname = "/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/input/starlink/DataSetForSaTE25/GrdStation/StarLink_DataSetForAgent25_5000_A.pkl"
    with open(paths_fname, 'rb') as file:
        data = pickle.load(file)

    with open(output_csv, "a") as results:
        print(",".join(TEST_HEADERS), file=results)
        
        for topo_idx in range(topo_num): # For every topology

            with open(AssetManager.tm_test_path(problem_path, topo_idx), 'rb') as f:
                tm = pickle.load(f)
            
            G = AssetManager.load_graph(problem_path, topo_idx)
        
            if test_all:
                test_size_per_topo = tm.shape[0]
            
            for tm_idx in tqdm(range(test_size_per_topo), desc=f"Computing topology {topo_idx}"):
                
                problem = Problem(G, tm[tm_idx], scale_factor=scale_factor)
                traffic_seed = problem.traffic_matrix.seed
                total_demand = problem.total_demand
                problem.name = str(tm_idx)
                pop = POP(
                            objective=Objective.get_obj_from_str(obj),
                            topo_idx=tm_idx,
                            num_subproblems=num_subproblems,
                            problem_path=problem_path,
                            split_method=split_method,
                            split_fraction=split_fraction,
                            algo_cls=algo_cls,
                            num_paths=num_paths,
                            edge_disjoint=edge_disjoint,
                            dist_metric=dist_metric,
                        )
                try:
                    pop.solve(problem)
                    result_line = TEST_PLACEHOLDER.format(
                        topo_idx,
                        tm_idx,
                        total_demand,
                        pop.obj_val,
                        pop.obj_val / total_demand,
                        pop.runtime,
                        )
                    print(result_line, file=results)
                except:
                    print("Error")




if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-subproblems",type = int,
                    choices = [1, 2, 4, 8, 16, 32, 64, 128],
                    nargs =  "+",
                    default = 16,
                    help = "Number of subproblems to use",)
    
    parser.add_argument("--split-methods", type = str,
                    choices= ["random", "means", "tailored", "skewed", "covs"],
                    nargs = "+",
                    default = "random",
                    help = "Split method to use")
    parser.add_argument("--split-fractions", type = float,
                    choices = [0, 0.25, 0.5, 0.75, 1.0],
                    nargs = "+",
                    default = 0,
                    help = "Split fractions to use")
    parser.add_argument("--algo-cls", type = str,
                    choices = ["PathFormulation", "TEAVAR"],
                    default = "PathFormulation",
                    action = AlgoClsAction,
                    help = "which underlying algorithm to benchmark with POP")

    
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
