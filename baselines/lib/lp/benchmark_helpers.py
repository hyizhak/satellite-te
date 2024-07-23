from collections import defaultdict
from glob import iglob

import argparse
import os
import sys

sys.path.append("..")

from lib.lp.algorithms import PathFormulation, TEAVAR
from lib.lp.algorithms.abstract_formulation import OBJ_STRS
from lib.lp.partitioning import FMPartitioning, SpectralClustering
from lib.lp.config import TOPOLOGIES_DIR, TM_DIR

PROBLEM_NAMES = [
    'B4.json',
    'UsCarrier.json',
    'Kdl.json',
    'ASN2k.json',
]
TM_MODELS = [
    "real",
    "toy",
]
SCALE_FACTORS = [1.0]

PATH_FORM_HYPERPARAMS = (4, True, "min-hop")
NCFLOW_HYPERPARAMS = {
    'B4.json': (4, True, 'min-hop', FMPartitioning, 3),
    'UsCarrier.json': (4, True, 'min-hop', FMPartitioning, 3),
    'Kdl.json': (4, True, 'min-hop', FMPartitioning, 3),
    'ASN2k.json': (4, True, 'min-hop', FMPartitioning, 3),
}

PROBLEM_NAMES_AND_TM_MODELS = [
    (prob_name, tm_model) for prob_name in PROBLEM_NAMES for tm_model in TM_MODELS
]

PROBLEMS = []
GROUPED_BY_PROBLEMS = defaultdict(list)
HOLDOUT_PROBLEMS = []
GROUPED_BY_HOLDOUT_PROBLEMS = defaultdict(list)

for problem_name in PROBLEM_NAMES:
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join(TOPOLOGIES_DIR, "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join(TOPOLOGIES_DIR, problem_name)
    for model in TM_MODELS:
        for tm_fname in iglob(
            "{}/{}/{}*_traffic-matrix.pkl".format(TM_DIR, model, problem_name)
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            PROBLEMS.append((problem_name, topo_fname, tm_fname))
        for tm_fname in iglob(
            "{}/holdout/{}/{}*_traffic-matrix.pkl".format(
                TM_DIR, model, problem_name
            )
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))

GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
for key, vals in GROUPED_BY_PROBLEMS.items():
    GROUPED_BY_PROBLEMS[key] = sorted(vals, 
        key=lambda x: int(x[-1].split('_')[-3]))

GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
    GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(vals, 
        key=lambda x: int(x[-1].split('_')[-3]))


# This should be called when `many_problems` is False
def get_problem(args):
    if (args.topo, args.tm_model, args.scale_factor) not in GROUPED_BY_PROBLEMS:
        raise Exception('Traffic matrices not found')
    topo_fname, tm_fname = GROUPED_BY_PROBLEMS[
        (args.topo, args.tm_model, args.scale_factor)
    ][args.slice]
    return [(args.topo, topo_fname, tm_fname)]


# This should be called when `many_problems` is True
def get_problems(args):
    problems = []
    for (
        (problem_name, tm_model, scale_factor),
        topo_and_tm_fnames,
    ) in GROUPED_BY_PROBLEMS.items():
        for topo_fname, tm_fname in topo_and_tm_fnames:
            if (
                ("all" in args.topos or problem_name in args.topos)
                and ("all" in args.tm_models or tm_model in args.tm_models)
                and ("all" in args.scale_factors or scale_factor in args.scale_factors)
            ):
                problems.append((problem_name, topo_fname, tm_fname))
    if not problems:
        raise Exception('Traffic matrices not found')
    return problems[args.slice_start:args.slice_stop]


class AlgoClsAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        if value == "PathFormulation":
            value = PathFormulation
        elif value == "TEAVAR":
            value = TEAVAR
        setattr(namespace, self.dest, value)


def check_gurobi_license():
    if not os.system('gurobi_cl --license'):
        return True
    else:
        return False


def get_args_and_problems(
    formatted_fname_template,
    additional_args=[],
    *,
    many_problems=True,
):
    if not check_gurobi_license():
        raise Exception("Gurobi license not found")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False)
    parser.add_argument(
        "--obj",
        type=str,
        choices=OBJ_STRS,
        required=True,
    )
    if many_problems:
        parser.add_argument(
            "--tm-models",
            type=str,
            choices=TM_MODELS + ["all"],
            nargs="+",
            default="all",
        )
        parser.add_argument(
            "--topos",
            type=str,
            choices=PROBLEM_NAMES + ["all"],
            nargs="+",
            default="all",
        )
        parser.add_argument(
            "--scale-factors",
            type=lambda x: x if x == "all" else float(x),
            choices=SCALE_FACTORS + ["all"],
            nargs="+",
            default="all",
        )
        parser.add_argument(
            "--slice-start", type=int, default = 28
        )
        parser.add_argument(
            "--slice-stop", type=int, default = 36 
        )
    else:
        parser.add_argument("--tm-model", type=str, choices=TM_MODELS, required=True)
        parser.add_argument("--topo", type=str, choices=PROBLEM_NAMES, required=True)
        parser.add_argument(
            "--scale-factor", type=float, choices=SCALE_FACTORS, required=True
        )
        parser.add_argument("--slice", type=int, required=True)
    for add_arg in additional_args:
        name_or_flags, kwargs = add_arg[0], add_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    args = parser.parse_args()
    if many_problems:
        slice_str = "all" # "slice_" + "_".join(str(i) for i in args.slices)
        formatted_fname_substr = formatted_fname_template.format(args.obj, slice_str)
        return args, formatted_fname_substr, get_problems(args)
    else:
        formatted_fname_substr = format_args_for_filename(
            formatted_fname_template, args, additional_args
        )
        return args, formatted_fname_substr, get_problem(args)

# Should only be used when `many_problems` is False
def format_args_for_filename(template, args, additional_args):
    slice_str = "slice_{}".format(args.slice)
    additional_info_str = "problem_{}-tm_model_{}-scale_factor_{}".format(
        args.topo, args.tm_model, args.scale_factor
    )
    args_as_dict = vars(args)
    for add_arg in additional_args:
        add_arg_formatted = add_arg[0][2:].replace("-", "_")
        additional_info_str += "-{}_{}".format(
            add_arg_formatted, args_as_dict[add_arg_formatted]
        )
    formatted_string = template.format(args.obj, slice_str + "_" + additional_info_str)

    return formatted_string


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
