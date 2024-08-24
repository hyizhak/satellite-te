import os
import sys
import pickle

_ROOT = f'{os.path.dirname(__file__)}/..'

# ========== Benchmarking arguments
# Input and output
ARG_PROBLEM_PATH = f'{_ROOT}/input/IridiumDataSet14day20sec_Int5'
SOLUTION_PATH = f'/data/projects/11003765/sate/input/lp_solutions'
ARG_TOPO_NUM = 10
ARG_OUTPUT_DIR = f'{_ROOT}/output'
ARG_OUTPUT_PREFIX = None
# Path forming parameters
ARG_PATH_NUM = 5
ARG_EDGE_DISJOINT = False
ARG_DIST_METIRC = "min-hop"

ARG_OBJ = "teal_total_flow"
ARG_LOSS = "kl_div"
ARG_SCALE_FACTOR = 1.0

ARG_TEST_TM_PER_TOPO = None

# ==========

TEST_HEADERS = [
            "topo_idx",
            "tm_idx",
            "total_demand",
            "obj_val",
            "ratio",
            "runtime",
        ]
TEST_PLACEHOLDER = ','.join(['{}' for _ in TEST_HEADERS])

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib import AssetManager

import logging
LOGGING_FORMAT = "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt="%d/%b/%Y %H:%M:%S",
)

OBJ_STRS = ["teal_total_flow", "rounded_total_flow", "total_flow", "teal_min_max_link_util"]
LOSS_STRS = ["kl_div", "wasserstein"]
SCALE_FACTORS = [1.0]

def update_output_path(args, model):
    if args.output_prefix is None:
        parent_basename = os.path.basename(os.path.dirname(args.problem_path))
        problem_basename = os.path.basename(args.problem_path)
        args.output_prefix = f'{parent_basename}_{problem_basename}_{model}'
    args.work_dir = os.path.join(args.output_dir, args.output_prefix)

def read_solutions(file_path, smoothing=0.1):
    solutions = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Load each solution sequentially
                sol = pickle.load(f)
                if smoothing > 0:
                    sol = sol * (1 - smoothing) + smoothing / sol.shape[-1]                    
                solutions.append(sol)
            except EOFError:
                # End of file reached
                break
    return solutions
