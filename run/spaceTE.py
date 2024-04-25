#! /usr/bin/env python
import argparse
import time
import os
import torch
import psutil
import GPUtil
import statistics
import pickle
import re

from _common import *
from pathlib import Path

from lib.spaceTE import DyToPEnv, DyToPActor, DyToP
from lib.data.starlink.orbit_params import OrbitParams
from lib.data.starlink.ism import InterShellMode as ISM



# ========== Benchmarking arguments
# Benchmarking targets
ARG_TRAIN = False
ARG_TEST = False


def benchmark(args):
    
    obj, problem_path = args.obj, args.problem_path
    
    train, test = args.train, args.test
    
    device = torch.device(
        f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    # ========== load hyperparameters
    # env hyper-parameters
    topo_num = args.topo_num
    
    trainval_tm_per_topo = args.train_tm_per_topo
    
    if trainval_tm_per_topo is None:
        train_all = True
        val_ratio = args.val_ratio
        train_size_per_topo = val_size_per_topo = None
    else:
        train_all = False
        val_ratio = None
        val_size_per_topo = args.train_tm_per_topo * args.val_ratio 
        train_size_per_topo = args.train_tm_per_topo - val_size_per_topo
        
    if args.test_tm_per_topo is None:
        test_all = True
        test_size_per_topo = None
    else:
        test_all = False
        test_size_per_topo = args.test_tm_per_topo
        
    # path form parameters
    path_num, edge_disjoint, dist_metric = args.path_num, args.edge_disjoint, args.dist_metric
    
        
    # actor hyper-parameters
    topo_gnn = args.topo_gnn
    rho = args.rho
    # training hyper-parameters
    lr = args.lr
    early_stop = args.early_stop
    epoch_num = args.epochs
    batch_size = args.bsz
    sample_num = args.samples
    admm_step_num = args.admm_steps
    # testing hyper-parameters
    num_failure = args.failures
    quantized = args.quantized
    compiled = args.compiled
    # output
    work_dir = args.work_dir

    # ========== init dytop env, actor, mode
    
    train_id = _dytop_train_id(
        topo_num = topo_num, train_sz=trainval_tm_per_topo, val_ratio=val_ratio,
        lr=lr, epoches=epoch_num, batch_sz=batch_size,
        coma_sample=sample_num,
        rho=rho, admm_step=admm_step_num
    )

    path = Path(problem_path)

    if len(path.parts) > 1 and path.parts[-3] == 'starlink':

        print('Starlink!')

        params = OrbitParams(
            GrdStationNum=222,
            # GrdStationNum=0,
            Offset5=4236,
            graph_node_num=8694,
            # graph_node_num=8472,
            isl_cap=200,
            uplink_cap=800,
            downlink_cap=800,
            ism=ISM.GRD_STATION,
            # ism=ISM.ISL
        )

        print(params)

        match = re.search(r'\d+', path.parts[-2])
        intensity = int(match.group())

        print(f'Loading Starlink data for intensity {intensity}')

        with open(os.path.join(problem_path, f'StarLink_DataSetForAgent{intensity}_5000_A.pkl'), 'rb') as file:
            data_part1 = pickle.load(file)

        with open(os.path.join(problem_path, f'StarLink_DataSetForAgent{intensity}_5000_B.pkl'), 'rb') as file:
            data_part2 = pickle.load(file)

        dataset = data_part1 + data_part2


        dytop_env = DyToPEnv(
            obj=obj,
            problem_path=problem_path,
            num_path=path_num,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            rho=rho,
            work_dir=work_dir,
            dataset=dataset,
            num_failure=num_failure,
            orbit_params=params,
            device=device)
        dytop_actor = DyToPActor(
            dytop_env=dytop_env,
            topo_gnn=topo_gnn,
            train_id=train_id,
            device=device)
        dytop = DyToP(
            dytop_env=dytop_env,
            dytop_actor=dytop_actor,
            lr=lr,
            early_stop=early_stop)
        
    else:

        print('Iridium...')

        dytop_env = DyToPEnv(
            obj=obj,
            # topo=topo,
            problem_path=problem_path,
            num_topo=topo_num,
            num_path=path_num,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            rho=rho,
            train_all=train_all, 
            val_ratio=val_ratio,
            train_size=train_size_per_topo, val_size=val_size_per_topo,
            test_all=test_all,
            test_size=test_size_per_topo,
            work_dir=work_dir,
            num_failure=num_failure,
            device=device)
        dytop_actor = DyToPActor(
            dytop_env=dytop_env,
            topo_gnn=topo_gnn,
            train_id=train_id,
            device=device)
        dytop = DyToP(
            dytop_env=dytop_env,
            dytop_actor=dytop_actor,
            lr=lr,
            early_stop=early_stop)

        
    

    # ========== train
    if train:
        train_log_path = os.path.join(
            AssetManager.train_log_dir(work_dir, create_dir=True),
            f'{train_id}.log'
        )
        train_log_hdl = logging.FileHandler(train_log_path, mode="w")
        train_log_hdl.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logging.getLogger().addHandler(train_log_hdl)

        cpu_usage = []
        memory_usage = []

        log_gpu = True if device.type == 'cuda' else False
        if log_gpu:
            gpu = GPUtil.getGPUs()[0]
            gpu_load = []
            gpu_memory_usage = []

        logging.info('Training starts')
        train_start_time = time.time()

        dytop.train(
            num_epoch=epoch_num,
            batch_size=batch_size,
            num_sample=sample_num
        )

        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)
        if log_gpu:
            gpu_load.append(gpu.load)
            gpu_memory_usage.append(gpu.memoryUtil)

        dytop.save_model()

        train_stop_time = time.time()
        
        logging.info('Training ends')
        
        logging.info(f'<Training Time> {train_stop_time - train_start_time} s')
        logging.info(f'<CPU Usage Mean> {statistics.mean(cpu_usage)}')
        logging.info(f'<CPU Usagee List> {cpu_usage}')
        logging.info(f'<Memory Usage Mean> {statistics.mean(memory_usage)}')
        logging.info(f'<Memory Usage List> {memory_usage}')
        if log_gpu:
            logging.info(f'<GPU Load Mean> {statistics.mean(gpu_load)}')
            logging.info(f'<GPU Load List> {gpu_load}')
            logging.info(f'<GPU Memory Usage Mean> {statistics.mean(gpu_memory_usage)}')
            logging.info(f'<GPU Memory Usage List> {gpu_memory_usage}')

        
        logging.getLogger().removeHandler(train_log_hdl)
        
    # ========== test  
    if test:
        test_id = train_id + f'_quantized-{quantized}_compiled-{compiled}'
        test_log_dir = AssetManager.test_log_dir(work_dir, create_dir=True)
        output_csv = os.path.join(test_log_dir, f'{test_id}.csv')

        with open(output_csv, 'w') as f:
            print(','.join(TEST_HEADERS), file=f)
            
        if not train:
            dytop.load_model(quantized, compiled)
        dytop.test(
            num_admm_step=admm_step_num,
            output_header=TEST_HEADERS,
            output_placeholder=TEST_PLACEHOLDER,
            output_csv=output_csv,
        )

    return


def _dytop_train_id(
        topo_num, train_sz, val_ratio,    # training data set
        lr, epoches, batch_sz,  # training hyperparams
        coma_sample,            # RL hyperparams
        rho, admm_step):        # ADMM hyperparams
    return f'spaceTE_topo-{topo_num}_tsz-{train_sz}_vr-{val_ratio}_' + \
    f'lr-{lr}_ep-{epoches}_bsz-{batch_sz}_' +\
    f'sample-{coma_sample}_' +\
    f'rho-{rho}_step-{admm_step}'
    
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    # problems arguments
    parser.add_argument("--problem-path", type=str, default=ARG_PROBLEM_PATH)
    parser.add_argument('--topo-num', type=int, default=ARG_TOPO_NUM)
    parser.add_argument("--scale-factor", type=float, default=ARG_SCALE_FACTOR, choices=SCALE_FACTORS, help="traffic matrix scale factor")
    parser.add_argument("--dry-run", dest="dry_run", default=False, action="store_true", help="list problems to run")
    parser.add_argument("--obj", type=str, default=ARG_OBJ, choices=OBJ_STRS, help="objective function")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    # output parameters
    parser.add_argument("--output-dir", type=str, default=ARG_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default=ARG_OUTPUT_PREFIX)

    parser.add_argument('--devid', type=int, default=0, help='GPU device id')

    # path form arguments
    parser.add_argument('--path-num', type=int, default=ARG_PATH_NUM)
    parser.add_argument('--edge-disjoint', type=bool, default=ARG_EDGE_DISJOINT)
    parser.add_argument('--dist-metric', type=str, default=ARG_DIST_METIRC)
    
    # env hyper-parameters
    parser.add_argument('--train-tm-per-topo', type=int)
    parser.add_argument('--test-tm-per-topo', type=int, default=ARG_TEST_TM_PER_TOPO)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    
    # actor hyper-parameters
    parser.add_argument('--topo-gnn', type=str, default="EdgeGAT", help='type of Topology GNN layer')
    parser.add_argument('--rho', type=float, default=1.0, help='rho in ADMM')

    # training hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')
    parser.add_argument('--bsz', type=int, default=32, help='batch size')
    parser.add_argument('--samples', type=int, default=5, help='number of COMA samples')
    parser.add_argument('--admm-steps', type=int, default=5, help='number of ADMM steps')
    parser.add_argument('--early-stop', type=bool, default=False, help='whether to stop early')

    # testing hyper-parameters
    parser.add_argument('--failures', type=int, default=0, help='number of edge failures')
    parser.add_argument('--quantized', action="store_true", help='whether to quantize the model')
    parser.add_argument('--compiled', action="store_true", help='whether to JIT-compile the model')

    args = parser.parse_args()

    update_output_path(args, 'spaceTE')

    if args.dry_run:
        print("Problem to run: {args.problem_path}\nWorking dir: {args.work_dir}")
    else:
        benchmark(args)
