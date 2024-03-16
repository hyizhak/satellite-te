#! /usr/bin/env python

import pickle
import argparse

import sys

sys.path.append("..")
sys.path.append("../..")

from benchmark_helpers import PATH_FORM_HYPERPARAMS
from lib.algorithms import TEAVAR
from lib.problem import Problem
from lib.graph_utils import check_feasibility


def run_teavar(args):
    topo_fname = args.topo_fname
    tm_fname = args.tm_fname

    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
    teavar = TEAVAR(
        availability=0.99,
        failure_scenarios=[[(0, 1)]],
        failure_probs=[0.9],
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        DEBUG=True,
    )
    teavar.solve(problem)
    for var in teavar._solver.model.getVars():
        print(var.varName, var.x)
    print("TEAVAR: {}".format(teavar.obj_val))
    sol_dict = teavar.sol_dict
    check_feasibility(problem, [sol_dict])
    with open("teavar-sol-dict.pkl", "wb") as w:
        pickle.dump(sol_dict, w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--topo_fname", type=str, required=True)
    parser.add_argument("--tm_fname", type=str, required=True)
    args = parser.parse_args()
    run_teavar(args)
