import pickle
import numpy as np
import shutil
import networkx as nx
import multiprocessing as mp
import os
from tqdm import tqdm

from ...asset import AssetManager
from . import multi_shell_graph as MSG
from .pathform import StarlinkPathFormer
from .ism import InterShellMode as ISM
from .user_node import generate_sat2user, generate_user2sat, generate_is_user

class StarlinkAdapter():
    
    def __init__(self, input_path, topo_file_template, file_volume, data_per_topo, ism:ISM, isl_cap, uplink_cap, downlink_cap, test_ratio, parallel=None):
        self.input_path = input_path
        self.input_topo_file_template = topo_file_template
        
        self.file_volume = file_volume
        self.data_per_topo = data_per_topo
        self.ism = ism
        self.isl_cap = isl_cap
        self.uplink_cap = uplink_cap
        self.downlink_cap = downlink_cap
        
        self.test_ratio = test_ratio
        
        self.parallel = parallel
        
    def adapt(self, output_path):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        args = []
        for i in self.file_volume:
            file_path = os.path.join(self.input_path, self.input_topo_file_template.format(i))
            args.append((file_path, self.data_per_topo, i, self.ism, self.uplink_cap, self.downlink_cap, self.isl_cap, self.test_ratio, output_path))
        
        with mp.Pool(self.parallel) as pool:
            pool.starmap(StarlinkAdapter._adapt_topo_file, args)
        
        
    @staticmethod
    def _adapt_topo_file(file_path, data_num, file_idx, ism, uplink_cap, downlink_cap, isl_cap, test_ratio, output_path):
        # ========= Orbit Shell Parameters =========
        OrbitNum1 = 72
        SatNum1   = 22

        OrbitNum2 = 72
        SatNum2   = 22

        OrbitNum3 = 58
        SatNum3   = 6

        OrbitNum4 = 36
        SatNum4   = 20

        GrdStationNum = 222
        
        # =========== Generate Intra-Shell Static Graph =========
        LatMat1 = [0 for i in range(OrbitNum1 * SatNum1)]
        LatMat2 = [0 for i in range(OrbitNum2 * SatNum2)]
        LatMat3 = [0 for i in range(OrbitNum3 * SatNum3)]
        LatMat4 = [0 for i in range(OrbitNum4 * SatNum4)]
        LatLimit = 90

        Offset1 = 0
        Offset2 = OrbitNum1 * SatNum1
        Offset3 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2
        Offset4 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2 + OrbitNum3 * SatNum3
        Offset5 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2 + OrbitNum3 * SatNum3 + OrbitNum4 * SatNum4
        
        satellite_num = Offset5

        G1, EMap1, E1 = MSG.Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, Offset1, LatLimit)
        G2, EMap2, E2 = MSG.Inter_Shell_Graph(OrbitNum2, SatNum2, LatMat2, Offset2, LatLimit)
        G3, EMap3, E3 = MSG.Inter_Shell_Graph(OrbitNum3, SatNum3, LatMat3, Offset3, LatLimit)
        G4, EMap4, E4 = MSG.Inter_Shell_Graph(OrbitNum4, SatNum4, LatMat4, Offset4, LatLimit)

        file = open(file_path, 'rb')

        train_num = int(data_num * (1 - test_ratio))

        for k in tqdm(range(data_num)):
            
            data = pickle.load(file)
            
            # 1. Save pathform metadata
            meta = StarlinkPathFormer.create_metadata(Offset5, GrdStationNum, data['InterShell_GrdRelay'], data['InterShell_ISL'])
            AssetManager.save_pathform_metadata_(output_path, file_idx, k, meta)
            
            # 2. Cerate the topology
            match ism:
                case ISM.GRD_STATION:
                    G_interShell = data['InterShell_GrdRelay']
                    E_inter = []
                    for SatIndex in range(len(G_interShell)):
                        E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset5)])
                        E_inter.append([int(G_interShell[SatIndex] + Offset5), SatIndex])
                    graph_node_num = Offset5 * 2 + GrdStationNum

                case _:
                    raise NotImplementedError()
                    ISL_interShell = data['InterShell_ISL']
                    E_inter = []
                    for SatIndex in range(len(ISL_interShell[0])):
                        E_inter.append([SatIndex, int(ISL_interShell[0][SatIndex] + Offset2)])
                        E_inter.append([int(ISL_interShell[0][SatIndex] + Offset2), SatIndex])
                    
                    for SatIndex in range(len(ISL_interShell[1])):
                        E_inter.append([int(SatIndex + Offset2), int(ISL_interShell[1][SatIndex] + Offset3)])
                        E_inter.append([int(ISL_interShell[1][SatIndex] + Offset3), int(SatIndex + Offset2)])
                    
                    for SatIndex in range(len(ISL_interShell[2])):
                        E_inter.append([int(SatIndex + Offset3), int(ISL_interShell[2][SatIndex] + Offset4)])
                        E_inter.append([int(ISL_interShell[2][SatIndex] + Offset4), int(SatIndex + Offset3)])
                    graph_node_num = Offset5 * 2
                    
            sat2user = generate_sat2user(satellite_num, GrdStationNum, ism)
            
            E = E1 + E2 + E3 + E4 + E_inter
            G = nx.DiGraph()
            G.add_nodes_from(range(graph_node_num))
            ## 1. Inter-satellite links
            for e in E:
                G.add_edge(e[0], e[1], capacity=isl_cap)
            ## 2. User-satellite links
            for i in range(satellite_num):
                # Uplink
                G.add_edge(sat2user(i), i, capacity=uplink_cap)
                # Downlink
                G.add_edge(i, sat2user(i), capacity=downlink_cap)
            ## 3. Inter ground station links
            for i in range(GrdStationNum):
                for j in range(GrdStationNum):
                    if i == j:
                        continue
                    G.add_edge(i + Offset5, j + Offset5, capacity=0)
                    G.add_edge(j + Offset5, i + Offset5, capacity=0)
                
            AssetManager.save_graph_(output_path, file_idx, k, G)

             # 3. Create traffic matrices
            demand_dict = {}
            
            flowset = data['FlowSet']
            for flow in flowset:
                src = sat2user(flow[0])
                dst = sat2user(flow[1])
                d = flow[2]
                
                outer = demand_dict.get(src, {})
                inner = outer.get(dst, 0)
                outer[dst] = inner + d
                demand_dict[src] = outer
            
            edge_list = []
            weight_list = []
            
            for src, inner in demand_dict.items():
                for dst, amount in inner.items():
                    edge_list.append([src, dst])
                    weight_list.append(amount)
                
            demand_matrix = {
                'size': graph_node_num,
                'edge_list': np.array(edge_list, dtype=np.uint16),
                'weight_list': np.array(weight_list, dtype=np.float32),
            }
                            
            if k < train_num:
                AssetManager.save_tm_train_separate_(output_path, k if file_idx == "A" else 5000 + k, 0, demand_matrix)
            elif k < data_num:
                AssetManager.save_tm_test_separate_(output_path, k - train_num if file_idx == "A" else 5000 + k - train_num, 0, demand_matrix)
            else:
                break;

        file.close()
            
        
        
    
    
    @staticmethod
    def matrix_from_tm_file(file_path)->np.ndarray:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Reconstruct the traffic matrix
        size = data['size']
        edge_list = data['edge_list']
        weight_list = data['weight_list']
        
        tm = np.zeros((size, size))
        for edge, weight in zip(edge_list, weight_list):
            tm[edge[0], edge[1]] = weight
        
        return tm
