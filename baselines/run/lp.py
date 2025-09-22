import os
import pickle
import traceback
from tqdm import tqdm
import sys
import time
import gc

sys.path.append("..")

from lib.lp.problem import Problem
from lib.lp.algorithms import TopFormulation, PathFormulation, Objective
from lib.lp.traffic_matrix import *
from lib.lp.graph_utils import update_from_sols, compute_used_caps
from lib.lp.algorithms.load_starlink import construct_from_edge,dict_to_numpy, pop_split, sol_dict_to_tensor

from _common import *

import argparse

INTENSITY = 100
MODE = 'ISL'
ONLINE = False
SIZE = 5000
CHUNK = 1000

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
    num_run = args.num_run
    chunk = args.chunk if num_run > args.chunk else num_run

    problem_path, topo_num, obj = args.problem_path, args.topo_num, args.obj
    
    logging.info("LP solver")
    if size == 5000:
        paths_fname = f".../input/starlink/DataSetForSaTE{intensity}/{mode}/StarLink_DataSetForAgent{intensity}_5000_B.pkl"
    elif size == 66:
        paths_fname = '.../input/iridium/new_form/Iridium_DataSetForAgent_15_60480.pkl'
    else:
        paths_fname = f'.../input/starlink/{mode}/StarLink_DataSetForAgent100_5000_Size{size}.pkl'

    for i in range(int(num_run/chunk)):

        with open(paths_fname, 'rb') as file:
            data = pickle.load(file)
        logging.info(f"Loaded data from {paths_fname}")

        data = data[i*chunk:(i+1)*chunk]
        logging.info(f'Data {i*chunk} to {(i+1)*chunk} selected')

        for topo_idx in tqdm(range(chunk), mininterval=600): # For every topology

            if POP_ratio != 1:
                data_ = data[topo_idx]
                tm = data_['tm']
                tm = dict_to_numpy(tm, mode, size)
                # num_paths = data_['path']
                num_paths = 5
                start_time = time.time()
                G = construct_from_edge(data_['graph'], mode, online, size)
                orig_caps = {
                    (u,v): data["capacity"]
                    for u,v,data in G.edges(data=True)
                }
                # logging.info('seperating tms')
                tms = pop_split(tm, POP_ratio)
                # logging.info('seperated tms')
                pre_time = time.time() - start_time

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
                    graph=G
                )

                total_used = defaultdict(float)

                for i in range(POP_ratio):
                    problem.traffic_matrix = GenericTrafficMatrix(problem, tms[i])
                    obj_val = pf.solve(problem)
                    total_demand = np.sum(tms[i])
                    demand += total_demand
                    # throughput += pf.obj_val

                    used = compute_used_caps(pf.solution, pf.get_paths())
                    for e, f in used.items():
                        total_used[e] += f

                    G = update_from_sols(G, pf.solution, pf.get_paths())

                    problem = Problem(G, scale_factor=scale_factor)     
                
                    pf = PathFormulation(
                        data=data_,
                        objective=Objective.get_obj_from_str(obj),
                        path_num=num_paths,
                        edge_disjoint=edge_disjoint,
                        dist_metric=dist_metric,
                        graph=G
                    )

                runtime = time.time() - start_time
                # logging.info(pf.buildtime)
                # logging.info(pf.runtime)
                utilizations = [
                    total_used[e] / orig_caps[e]
                    for e in orig_caps if orig_caps[e] > 1
                ]
                final_mlu = max(utilizations)
                result_line = TEST_PLACEHOLDER.format(
                    topo_idx,
                    demand,
                    final_mlu,
                    throughput / demand,
                    pre_time + runtime/POP_ratio,
                )
                # logging.info(pre_time)
                # logging.info(runtime)
                try:
                    # Attempt to open the file in 'x' mode
                    with open(args.output_csv, 'x') as f:
                        print(result_line, file=f)
                except FileExistsError:
                    # If the file already exists, open it in append mode
                    with open(args.output_csv, 'a') as f:
                        print(result_line, file=f)
        
            else:

                data_ = data[topo_idx]
                tm = data_['tm']
                tm = dict_to_numpy(tm, mode, size)
                # num_paths = data_['path']
                num_paths = 5

                start_time = time.time()    
                G = construct_from_edge(data_['graph'], mode, online, POP_ratio, size)
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
                        graph=G
                    )
                obj_val = pf.solve(problem)

                solution = sol_dict_to_tensor(pf.solution)

                total_demand = problem.total_demand
                result_line = TEST_PLACEHOLDER.format(
                    topo_idx,
                    total_demand,
                    obj_val,
                    obj_val / total_demand,
                    time.time() - start_time,
                )
                try:
                    # Attempt to open the file in 'x' mode
                    with open(args.output_csv, 'x') as f:
                        print(result_line, file=f)
                except FileExistsError:
                    # If the file already exists, open it in append mode
                    with open(args.output_csv, 'a') as f:
                        print(result_line, file=f)

                with open(args.sol_dir, 'ab') as f:
                    pickle.dump(solution, f)

                # Explicitly delete large objects and collect garbage
                del pf, solution, G, problem, tm, data_
                gc.collect()

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
    parser.add_argument("--num-run", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=CHUNK)

    # output paramenters
    parser.add_argument("--output-dir", type=str, default=ARG_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str)
    
    # path form arguments
    parser.add_argument('--path-num', type=int, default=ARG_PATH_NUM)
    parser.add_argument('--edge-disjoint', type=bool, default=ARG_EDGE_DISJOINT)
    parser.add_argument('--dist-metric', type=str, default=ARG_DIST_METIRC)
    
    parser.add_argument('--test-tm-per-topo', type=int, default=ARG_TEST_TM_PER_TOPO)
    
    args = parser.parse_args()

    args.sol_dir = os.path.join(args.output_dir, f"Gurobi_size-{args.size}_mode-{args.mode}_intensity-{args.intensity}_volume-{args.num_run}_solutions.pkl")
    
    update_output_path(args, f"lp_obj-{args.obj}_{args.intensity}_{args.size}_{args.mode}_Online-{args.online}_POP-{args.POP_ratio}")
    

    if args.dry_run:
        print("Problem to run: {args.problem_path}\nWorking dir: {args.work_dir}")
    else:
        benchmark(args)
