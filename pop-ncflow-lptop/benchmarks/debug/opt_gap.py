#! /usr/bin/env python

import sys
import argparse

sys.path.append("..")
sys.path.append("../..")

from lib.partitioning.hard_coded_partitioning import HardCodedPartitioning
from lib.algorithms import NcfEpi, Objective, PathFormulation, POP
from lib.problems import OptGapC1, OptGapC2, OptGapC3, OptGapC4, BadForNCFlow, BadForPOP
from lib.graph_utils import check_feasibility

OBJ_STR = "total_flow"


def get_problem(problem_arg):
    if problem_arg == "OptGapC1":
        return OptGapC1()
    elif problem_arg == "OptGapC2":
        return OptGapC2()
    elif problem_arg == "OptGapC3":
        return OptGapC3()
    elif problem_arg == "OptGapC4":
        return OptGapC4()
    elif problem_arg == "BadForNCFlow":
        return BadForNCFlow()
    elif problem_arg == "BadForPOP":
        return BadForPOP()


def print_and_check(label, algo):
    print("{}: {}".format(label, algo.obj_val))
    sol_dict = algo.sol_dict
    check_feasibility(problem, [sol_dict])


def get_partition_vector(problem_name):
    if problem_name == "optgapc1":
        return [0, 0, 0, 1, 1, 1]
    elif problem_name == "optgapc2":
        return [0, 0, 1, 1, 2, 2, 3, 3]
    elif problem_name == "optgapc3":
        return [0, 0, 0, 0, 1, 1, 1]
    elif problem_name == "optgapc4":
        return [0, 0, 0, 1, 1, 1]
    elif problem_name == "BadForNCFlow":
        return [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    elif problem_name == "BadForPOP":
        return [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    else:
        raise Exception("Unknown problem name: {}".format(problem_name))


def run_ncflow(problem):
    partition_vector = get_partition_vector(problem.name)
    partitioner = HardCodedPartitioning(partition_vector)
    ncflow = NcfEpi(
        objective=Objective.get_obj_from_str(OBJ_STR),
        num_paths=1,
        edge_disjoint=True,
        dist_metric="inv-cap",
        DEBUG=True,
        VERBOSE=True,
    )
    ncflow.solve(problem, partitioner)
    return ncflow


def run_pop(problem):
    pop = POP(
        objective=Objective.get_obj_from_str(OBJ_STR),
        num_subproblems=2,
        split_fraction=0.0,
        split_method="random",
        algo_cls=PathFormulation,
        num_paths=1
    )
    pop.solve(problem)
    return pop


def run_pf(problem):
    pf = PathFormulation.get_pf_for_obj(Objective.get_obj_from_str(OBJ_STR), 4)
    pf.solve(problem)
    return pf


def run_optgap(problem):
    pf = run_pf(problem)
    ncflow = run_ncflow(problem)
    pop = run_pop(problem)
    print_and_check("PathFormulation", pf)
    print_and_check("NCFlow", ncflow)
    print_and_check("POP", pop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        choices=["OptGapC1", "OptGapC2", "OptGapC3", "OptGapC4", "BadForNCFlow", "BadForPOP"],
        required=True,
    )
    args = parser.parse_args()
    problem = get_problem(args.problem)
    run_optgap(problem)
