import os
import sys

_ROOT = f'{os.path.dirname(__file__)}/..'

# ========== Benchmarking arguments
# Input and output
ARG_PROBLEM_PATH = f'{_ROOT}/input/IridiumDataSet14day20sec_Int5'
ARG_TOPO_NUM = 10
ARG_OUTPUT_DIR = f'{_ROOT}/output'
ARG_OUTPUT_PREFIX = None
# Path forming parameters
ARG_PATH_NUM = 5
ARG_EDGE_DISJOINT = False
ARG_DIST_METIRC = "min-hop"

ARG_OBJ = "total_flow"
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

OBJ_STRS = ["total_flow", "min_max_link_util"]
SCALE_FACTORS = [1.0]

def update_output_path(args, model):
    if args.output_prefix is None:
        args.output_prefix = os.path.basename(args.problem_path) + f'_{model}'
    args.work_dir = os.path.join(args.output_dir, args.output_prefix)
