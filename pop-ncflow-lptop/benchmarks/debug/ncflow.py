#! /usr/bin/env python

import pickle
import argparse
import numpy as np

import sys

sys.path.append("..")
sys.path.append("../..")

from benchmark_helpers import NCFLOW_HYPERPARAMS
from lib.algorithms.abstract_formulation import OBJ_STRS
from lib.algorithms import NcfEpi, Objective
from lib.problem import Problem
from lib.graph_utils import check_feasibility


def run_ncflow(args):
    obj = args.obj
    topo_fname = args.topo_fname
    tm_fname = args.tm_fname
    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    (
        num_paths,
        edge_disjoint,
        dist_metric,
        partition_cls,
        num_parts_scale_factor,
    ) = NCFLOW_HYPERPARAMS[problem.name]
    num_partitions_to_set = num_parts_scale_factor * int(np.sqrt(len(problem.G.nodes)))
    partitioner = partition_cls(num_partitions_to_set)

    ncflow = NcfEpi(
        objective=Objective.get_obj_from_str(obj),
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        DEBUG=True,
        VERBOSE=False,
    )
    ncflow.solve(problem, partitioner)
    print("{}: {}".format(obj, ncflow.obj_val))
    sol_dict = ncflow.sol_dict
    check_feasibility(problem, [sol_dict])
    with open("ncflow-sol-dict.pkl", "wb") as w:
        pickle.dump(sol_dict, w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj",
        type=str,
        choices=OBJ_STRS,
        required=True,
    )
    parser.add_argument("--topo_fname", type=str, required=True)
    parser.add_argument("--tm_fname", type=str, required=True)
    args = parser.parse_args()
    run_ncflow(args)
