#! /usr/bin/env python

from dytop_helper import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS

import os
import sys

import torch

sys.path.append('..')

from lib.dytop_env import DyToPEnv
from lib.dytop_actor import DyToPActor
from lib.dytop_model import DyToP


TOP_DIR = "dytop-logs"
MODEL_DIR = "dytop-models"
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "scale_factor",
    "tm_model",
    "total_demand",
    "algo",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
    "objective",
    "obj_val",
    "runtime",
]

OUTPUT_CSV_TEMPLATE = "dytop-{}-{}.csv"


def benchmark(problems, output_csv, arg):

    num_path, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
    # obj, topo = args.obj, args.topo
    obj, constellation = args.obj, args.constellation
    model_save = args.model_save
    device = torch.device(
        f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    # ========== load hyperparameters
    # env hyper-parameters
    train_size = [args.slice_train_start, args.slice_train_stop]
    val_size = [args.slice_val_start, args.slice_val_stop]
    test_size = [args.slice_test_start, args.slice_test_stop]
    # actor hyper-parameters
    topo_gnn = args.topo_gnn
    num_layer = args.layers
    rho = args.rho
    # training hyper-parameters
    lr = args.lr
    early_stop = args.early_stop
    num_epoch = args.epochs
    batch_size = args.bsz
    num_sample = args.samples
    num_admm_step = args.admm_steps
    # testing hyper-parameters
    num_failure = args.failures

    # ========== init dytop env, actor, model
    dytop_env = DyToPEnv(
        obj=obj,
        # topo=topo,
        constellation=constellation,
        problems=problems,
        num_path=num_path,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        rho=rho,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_failure=num_failure,
        device=device)
    dytop_actor = DyToPActor(
        dytop_env=dytop_env,
        topo_gnn=topo_gnn,
        num_layer=num_layer,
        model_dir=MODEL_DIR,
        model_save=model_save,
        device=device)
    dytop = DyToP(
        dytop_env=dytop_env,
        dytop_actor=dytop_actor,
        lr=lr,
        early_stop=early_stop)

    # ========== train and test
    dytop.train(
        num_epoch=num_epoch,
        batch_size=batch_size,
        num_sample=num_sample)
    dytop.test(
        num_admm_step=num_admm_step,
        output_header=HEADERS,
        output_csv=output_csv,
        output_dir=TOP_DIR)

    return


if __name__ == '__main__':

    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    args, output_csv, problems = get_args_and_problems(OUTPUT_CSV_TEMPLATE)

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems, output_csv, args)
