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
import random
import numpy as np

from lib.spaceTE import SaTEEnv, SaTEActor, SaTE
from lib.data.starlink.orbit_params import OrbitParams
from lib.data.starlink.ism import InterShellMode as ISM



# ========== Benchmarking arguments
# Benchmarking targets
ARG_TRAIN = False
ARG_TEST = False


def benchmark(args):

    set_seed(42)
    
    obj, problem_path = args.obj, args.problem_path
    
    train, test, admm_test, supervised, panelized, dummy = args.train, args.test, args.admm_test, args.supervised, args.penalized, args.dummy

    print(f'Running SaTE with train={train}, supervised={supervised}, panelized={panelized}, test={test}, admm_test={admm_test}, dummy_path={dummy}')
    
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
    layers = args.layers
    decoder_type = args.decoder
    # training hyper-parameters
    lr = args.lr
    early_stop = args.early_stop
    epoch_num = args.epochs
    batch_size = args.bsz
    sample_num = args.samples
    admm_step_num = args.admm_steps
    flow_lambda = args.flow_lambda
    # testing hyper-parameters
    num_failure = args.failures
    quantized = args.quantized
    compiled = args.compiled
    model_path = args.model_path
    # output
    work_dir = args.work_dir

    # ========== init SaTE env, actor, mode
    
    train_id = _sate_train_id(
        obj=obj, train_sz=trainval_tm_per_topo, val_ratio=val_ratio,
        lr=lr, epoches=epoch_num, batch_sz=batch_size, supervised=supervised, penalized=panelized, flow_lambda=flow_lambda, dummy=dummy,
        coma_sample=sample_num, layers=layers, decoder=decoder_type,
        rho=rho, admm_step=admm_step_num,
    )

    params = None

    path = Path(problem_path)

    if len(path.parts) > 1 and path.parts[-3] == 'starlink':

        print('Starlink!')

        pattern_4500 = r'^DataSetForSaTE.*'

        if re.match(pattern_4500, path.parts[-2]) is not None or path.parts[-2] == 'mixed':

            if path.parts[-1] == 'GrdStation':

                params = OrbitParams(
                    GrdStationNum=222,
                    Offset5=4236,
                    graph_node_num=8694,
                    isl_cap=200,
                    uplink_cap=800,
                    downlink_cap=800,
                    ism=ISM.GRD_STATION,
                )

            elif path.parts[-1] == 'ISL':

                params = OrbitParams(
                    GrdStationNum=0,
                    Offset5=4236,
                    graph_node_num=8472,
                    isl_cap=200,
                    uplink_cap=800,
                    downlink_cap=800,
                    ism=ISM.ISL,
                )

            print(params)

            if path.parts[-2] == 'mixed':

                print('Loading mixed data')

                with open(os.path.join(problem_path, 'StarLink_DataSetForAgent_mixed_4000.pkl'), 'rb') as file:
                    dataset = pickle.load(file)

                if supervised:

                    label = []
                    for intensity in [25, 50, 75, 100]:
                        sol_dir = os.path.join(SOLUTION_PATH, f'Gurobi_size-5000_mode-{path.parts[-1]}_intensity-{intensity}_volume-1000_solutions.pkl')
                        label.extend(read_solutions(sol_dir))
                    
                    dataset = dataset[:len(label)]

                    dataset = (dataset, label)

            else:

                match = re.search(r'\d+', path.parts[-2])
                intensity = int(match.group())

                print(f'Loading Starlink data for intensity {intensity}')

                with open(os.path.join(problem_path, f'StarLink_DataSetForAgent{intensity}_5000_B.pkl'), 'rb') as file:
                    dataset = pickle.load(file)

                if supervised:
                    sol_dir = os.path.join(SOLUTION_PATH, f'Gurobi_size-5000_mode-{path.parts[-1]}_intensity-{intensity}_volume-1000_solutions.pkl')
                    label = read_solutions(sol_dir)
                    dataset = dataset[:len(label)]

                    dataset = (dataset, label)

            
        else:

            match = re.search(r'\d+', path.parts[-2])
            size = int(match.group())

            print(f'Loading reduced Starlink data for size {size}')

            match size:
                case 176:
                    reduced = 18
                case 500:
                    reduced = 8
                case 528:
                    reduced = 6
                case 1500:
                    reduced = 2

            if path.parts[-1] == 'GrdStation':

                params = OrbitParams(
                    GrdStationNum=222,
                    Offset5=round(2 * 22 * 72 / reduced),
                    graph_node_num=round(2 * 22 * 72 / reduced) * 2 + 222,
                    isl_cap=200,
                    uplink_cap=800,
                    downlink_cap=800,
                    ism=ISM.GRD_STATION,
                )

            elif path.parts[-1] == 'ISL':

                params = OrbitParams(
                    GrdStationNum=0,
                    Offset5=round(2 * 22 * 72 / reduced),
                    graph_node_num=round(2 * 22 * 72 / reduced) * 2,
                    isl_cap=200,
                    uplink_cap=800,
                    downlink_cap=800,
                    ism=ISM.ISL,
                )

            print(params)

            with open(os.path.join(problem_path, f'StarLink_DataSetForAgent100_5000_Size{size}.pkl'), 'rb') as file:
                dataset = pickle.load(file)

            if supervised:
                sol_dir = os.path.join(SOLUTION_PATH, f'Gurobi_size-{size}_mode-{path.parts[-1]}_intensity-100_volume-5000_solutions.pkl')
                label = read_solutions(sol_dir)
                dataset = dataset[:len(label)]

                dataset = (dataset, label)

    else:

        print('Iridium...')
        
        intensity = path.parts[-1].split("_")[1]

        print(f'Loading Iridium data for intensity {intensity}')

        with open(os.path.join(problem_path, f'Iridium_DataSetForAgent_{intensity}_60480.pkl'), 'rb') as file:
            dataset = pickle.load(file)

        if supervised:
            sol_dir = os.path.join(SOLUTION_PATH, f'Gurobi_size-66_mode-NA_intensity-{intensity}_volume-5000_solutions.pkl')
            label = read_solutions(sol_dir)
            dataset = dataset[:len(label)]

            dataset = (dataset, label)


    sate_env = SaTEEnv(
        obj=obj,
        problem_path=problem_path,
        num_path=path_num,
        dummy_path=dummy,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        rho=rho,
        work_dir=work_dir,
        dataset=dataset,
        supervised=supervised,
        penalized=panelized,
        flow_lambda=flow_lambda,
        orbit_params=params,
        num_failure=num_failure,
        device=device)
    sate_actor = SaTEActor(
        sate_env=sate_env,
        topo_gnn=topo_gnn,
        layers=layers,
        decoder_type=decoder_type,
        train_id=train_id,
        device=device)
    sate = SaTE(
        sate_env=sate_env,
        sate_actor=sate_actor,
        lr=lr,
        supervised=supervised,
        penalized=panelized,
        early_stop=early_stop)

        

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    logging.info(f'<Model Parameters> {count_parameters(sate_actor)}')
    

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

        sate.train(
            num_epoch=epoch_num,
            batch_size=batch_size,
            num_sample=sample_num
        )

        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)
        if log_gpu:
            gpu_load.append(gpu.load)
            gpu_memory_usage.append(gpu.memoryUtil)

        sate.save_model()

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
        test_id = train_id + f'_step-{admm_step_num}_failures-{num_failure}_admm-test-{admm_test}'
        test_log_dir = AssetManager.test_log_dir(work_dir, create_dir=True)
        output_csv = os.path.join(test_log_dir, f'{test_id}.csv')

        with open(output_csv, 'w') as f:
            print(','.join(TEST_HEADERS), file=f)
            
        if not (train or admm_test):
            sate.load_model(quantized, compiled, model_path)
        sate.test(
            num_admm_step=admm_step_num,
            output_header=TEST_HEADERS,
            output_placeholder=TEST_PLACEHOLDER,
            output_csv=output_csv,
            admm_test=admm_test
        )

    return


def _sate_train_id(
        obj, train_sz, val_ratio,    # training data set
        lr, epoches, batch_sz, supervised, penalized, flow_lambda, dummy, # training hyperparams
        coma_sample, layers, decoder,            # hyperparams
        rho, admm_step):        # ADMM hyperparams
    if supervised:
        train_mode = 'supervised'
    elif penalized:
        train_mode = 'penaltized-optimization'
    else:
        train_mode = 'RL'
    return f'spaceTE_{train_mode}_' + \
    f'ep-{epoches}_' + f'dummy-path-{dummy}_' \
    f'flow-lambda-{flow_lambda}_' + f'layers-{layers}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
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

    parser.add_argument("--admm-test", action="store_true")

    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--penalized", action="store_true")
    
    # output parameters
    parser.add_argument("--output-dir", type=str, default=ARG_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default=ARG_OUTPUT_PREFIX)

    parser.add_argument('--devid', type=int, default=0, help='GPU device id')

    # path form arguments
    parser.add_argument('--path-num', type=int, default=ARG_PATH_NUM)
    parser.add_argument('--dummy', action="store_true")
    parser.add_argument('--edge-disjoint', type=bool, default=ARG_EDGE_DISJOINT)
    parser.add_argument('--dist-metric', type=str, default=ARG_DIST_METIRC)
    
    # env hyper-parameters
    parser.add_argument('--train-tm-per-topo', type=int)
    parser.add_argument('--test-tm-per-topo', type=int, default=ARG_TEST_TM_PER_TOPO)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    
    # actor hyper-parameters
    parser.add_argument('--topo-gnn', type=str, default="EdgeGAT", help='type of Topology GNN layer')
    parser.add_argument('--rho', type=float, default=1.0, help='rho in ADMM')
    parser.add_argument('--layers', type=int, default=0, help='number of hidden layers in Topolohy GNN')
    parser.add_argument('--decoder', type=str, default="linear", help='type of decoder')

    # training hyper-parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--bsz', type=int, default=4, help='batch size')
    parser.add_argument('--samples', type=int, default=200, help='number of COMA samples')
    parser.add_argument('--admm-steps', type=int, default=5, help='number of ADMM steps')
    parser.add_argument('--early-stop', type=bool, default=False, help='whether to stop early')
    parser.add_argument('--flow-lambda', type=float, default=25, help='multiplier for total flow')

    # testing hyper-parameters
    parser.add_argument('--failures', type=float, default=0, help='number of edge failures (%)')
    parser.add_argument('--quantized', action="store_true", help='whether to quantize the model')
    parser.add_argument('--compiled', action="store_true", help='whether to JIT-compile the model')
    parser.add_argument('--model-path', type=str, default=None, help='path to the model to load')

    args = parser.parse_args()

    update_output_path(args, 'spaceTE')

    if args.dry_run:
        print("Problem to run: {args.problem_path}\nWorking dir: {args.work_dir}")
    else:
        benchmark(args)
