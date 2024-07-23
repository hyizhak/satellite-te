import os
import shutil
import pickle
import argparse
import networkx as nx
import copy
import numpy as np
import multiprocessing as mp

ROOT = f'{os.path.dirname(os.path.abspath(__file__))}/../..'

# ========== Program Arguments
# Input
ARG_INPUT_PATH = os.path.join(ROOT, 'input/raw/IridiumDataSet14day20sec_Int5')
ARG_INPUT_START_DAY = 0
ARG_INPUT_STOP_DAY = 14
ARG_INPUT_TM_SKIP = 270
# Output
ARG_OUTPUT_PATH = os.path.join(ROOT, 'input')
ARG_OUTPUT_PREFIX = None


# Add grandparent path in to import path
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from lib import AssetManager

class IridiumAdapter():
    
    DEMANDS_TMP_DIR = '.tm_tmp'
    DAY_SEC = 60 * 60 * 24
    
    def __init__(self,
        input_path:str,
        day_file_template:str,
        start_day:int, stop_day:int,
        tm_skip:int,
        
        topo_cycle_sec:int, period_sec:int,

        output_path:str,
        output_prefix:str,

        parallel:int,
        test_ratio:float,
    ):
        
        self._day_file_template = day_file_template
        
        self.topo_cycle_sec = topo_cycle_sec
        self.period_sec = period_sec
        if (self.DAY_SEC % period_sec != 0):
            raise ValueError('DAY_SEC must be divisible by period_sec')
        self.day_period_num = self.DAY_SEC // self.period_sec
        if (topo_cycle_sec % period_sec != 0):
            raise ValueError('topo_cycle_sec must be divisible by period_sec')
        self.topo_num = topo_cycle_sec // period_sec

        self.input_path = input_path
        self.start_day = start_day
        self.stop_day = stop_day

        self.skip_data = tm_skip
        
        self.prefix = output_prefix
        self.output_path = os.path.join(output_path, self.prefix)
        
        self.satellite_num = self._retrieve_satellite_num()
        self.node_num = self.satellite_num * 2
    
        self.parallel = parallel

        self.test_ratio = test_ratio
        
    def _user_node(self, satellite_idx):
        return _user_node(satellite_idx, self.satellite_num)
        
    def _retrieve_satellite_num(self):
        drec = self._load_day(self.start_day)
        return len(drec['topology_list'][0][0])
    

    def _get_day_file_path(self, day):
        return os.path.join(self.input_path, self._day_file_template.format(day))


    def _load_day(self, day):
        fpath = self._get_day_file_path(day)
        if (os.path.exists(fpath) == False):
            raise FileNotFoundError(fpath)
        return _load_file(fpath)
    
        
    def _day_topo_start(self, day):
        return ((day - self.start_day) * self.day_period_num) % self.topo_num

        
    def create_topologies(self):

        drec = self._load_day(self.start_day)

        topos = drec['topology_list'][:self.topo_num]
        isl_cap = drec['isl_cap']
        uplink_cap = drec['unplink_cap']
        downlink_cap = drec['downlink_cap']
                
        # Nodes [0, self.satellite_num) are satellite nodes. Nodes [self.satellite_num, self.satellite_num * 2) are user nodes.
        template = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": n} for n in range(self.satellite_num * 2)],
        }
        
        template_G = nx.DiGraph()
        template_G.add_nodes_from(range(self.satellite_num * 2))
        
        for i, topo in enumerate(topos):
            assert len(topo[0]) == self.satellite_num
            
            G = copy.deepcopy(template_G)
            
            # 1. User-satellite links
            for j in range(self.satellite_num):
                # Uplink
                G.add_edge(self._user_node(j), j, capacity=uplink_cap)
                # Downlink
                G.add_edge(j, self._user_node(j), capacity=downlink_cap)
                
            # 2. Inter-satellite links
            for l in topo[2]:
                # Do some checks here
                s, t = l[0], l[1]
                assert s < self.satellite_num and t < self.satellite_num and \
                topo[0][s][t] == 1 and topo[1][s][t] != -1
                G.add_edge(s, t, capacity=isl_cap)
                
            # Save the graph
            AssetManager.save_graph_(self.output_path, i, G)
                
    
    def create_demands(self):

        demand_tmp_path = os.path.join(self.output_path, self.DEMANDS_TMP_DIR)
        if os.path.exists(demand_tmp_path):
            shutil.rmtree(demand_tmp_path)
        os.makedirs(demand_tmp_path)

        # Skip data
        total_data = (self.stop_day - self.start_day) * self.day_period_num
        # Ensure every topology has the same amount of data
        total_skip = self.skip_data + ((total_data - self.skip_data) % self.topo_num)

        def map():
            args = []
            for day in range(self.start_day, self.stop_day):
                fpath = self._get_day_file_path(day)
                topo_num = self.topo_num
                topo_start = self._day_topo_start(day)
                to_skip = total_skip - (day - self.start_day) * self.day_period_num
                if to_skip <= 0:
                    skip = 0
                else:
                    skip = min(to_skip, self.day_period_num)
                day_total_data = self.day_period_num
                day_output_path = os.path.join(demand_tmp_path, str(day))
                os.makedirs(day_output_path)
                args.append((fpath, self.satellite_num, topo_num, topo_start, skip, day_total_data, day_output_path))

            with mp.Pool(self.parallel) as pool:
                day_demands_path = pool.starmap(_demand_mapper, args)

            return day_demands_path
        
        def reduce(day_demands_path):
            args = []
            node_num = self.node_num
            for topo_idx in range(self.topo_num):
                train_out_path = AssetManager.tm_train_path(self.output_path, topo_idx, True)
                test_out_path = AssetManager.tm_test_path(self.output_path, topo_idx, True)
                args.append((node_num, day_demands_path, topo_idx, train_out_path, test_out_path, self.test_ratio))

            with mp.Pool(self.parallel) as pool:
                pool.starmap(_demand_reducer, args)

        map_result_dirs = map()
        reduce(map_result_dirs)

        shutil.rmtree(demand_tmp_path)
        
        return
    

    def adapt(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        self.create_topologies()
        self.create_demands()


def _load_file(file_path):
    with open(file_path, 'rb') as f:
        pkl = pickle.load(f)
        
    return {
        'flowset_list': pkl[0],
        'topology_list': pkl[1],
        'isl_cap': pkl[2],
        'unplink_cap': pkl[3],
        'downlink_cap': pkl[4]
    }

def _user_node(satellite_idx, satellite_num):
    return satellite_idx + satellite_num

def _demand_mapper(file_path, sat_num, topo_num, topo_start, skip_data_num, total_data_num, output_path):
    dflow = _load_file(file_path)['flowset_list']
    assert total_data_num == len(dflow)

    topo_skip_start = (topo_start + skip_data_num) % topo_num
    data_num_remainder = (total_data_num - skip_data_num) % topo_num
    
    def topo_has_remainder(topo_idx):
        return (topo_idx + topo_num - topo_skip_start) % topo_num < data_num_remainder

    demands = []
    for topo_idx in range(topo_num):
        topo_data_num = (total_data_num - skip_data_num) // topo_num
        if topo_has_remainder(topo_idx):
            topo_data_num += 1
        demands.append(np.zeros((topo_data_num, sat_num * 2, sat_num * 2)))

    for i in range(skip_data_num, total_data_num):
        flows = dflow[i]
        topo_idx = (i + topo_start) % topo_num
        data_idx = (i - skip_data_num) // topo_num
        for f in flows:
            src = _user_node(f[0], sat_num)
            assert len(f[1]) == 1
            dst = _user_node(f[1][0], sat_num)
            demand = f[2]
            demands[topo_idx][data_idx][src][dst] += demand

    del dflow

    for i in range(topo_num):
        with open(os.path.join(output_path, f'{i}.pkl'), 'wb') as f:
            pickle.dump(demands[i], f)
    
    return output_path


def _demand_reducer(node_num, map_result_dirs, topo_idx, train_output_path, test_output_path, test_ratio):
    demands = np.zeros((0, node_num, node_num))
    for d in map_result_dirs:
        with open(os.path.join(d, f'{topo_idx}.pkl'), 'rb') as f:
            demands = np.concatenate((demands, pickle.load(f)), axis=0)

    train_num = int(demands.shape[0] * (1 - test_ratio))

    with open(train_output_path, 'wb') as f:
        pickle.dump(demands[:train_num], f)
    
    with open(test_output_path, 'wb') as f:
        pickle.dump(demands[train_num:], f)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Data adapter for Iridium')
    
    parser.add_argument('--input-path', type=str, default=ARG_INPUT_PATH)
    parser.add_argument('--input-start-day', type=int, default=ARG_INPUT_START_DAY)
    parser.add_argument('--input-stop-day', type=int, default=ARG_INPUT_STOP_DAY)
    parser.add_argument('--input-tm-skip', type=int, default=ARG_INPUT_TM_SKIP)
    parser.add_argument('--input-day-file-template', type=str, default='Iridium_DataSetForAgent_Day{}.pkl')

    parser.add_argument('--topo-cycle-sec', type=int, default=100*60, help="After how many seconds the same topology appears again")
    parser.add_argument('--period-sec', type=int, default=20, help="How long does a period (data point) last")

    parser.add_argument('--output-path', type=str, default=ARG_OUTPUT_PATH)
    parser.add_argument('--output-prefix', type=str, default=ARG_OUTPUT_PREFIX)

    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--test-ratio', type=float, default=0.15)

    args = parser.parse_args()

    if args.output_prefix is None:
        args.output_prefix = os.path.basename(args.input_path)

    adapter = IridiumAdapter(
        input_path=args.input_path, 
        day_file_template=args.input_day_file_template,
        start_day=args.input_start_day, 
        stop_day=args.input_stop_day,
        tm_skip=args.input_tm_skip,
        topo_cycle_sec=args.topo_cycle_sec, 
        period_sec=args.period_sec, 
        output_path=args.output_path,
        output_prefix=args.output_prefix,
        parallel=args.parallel,
        test_ratio=args.test_ratio
    )
    
    adapter.adapt()

    