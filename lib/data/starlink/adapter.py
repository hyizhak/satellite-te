import pickle
import numpy as np
import shutil
import networkx as nx
import multiprocessing as mp
import os
from tqdm import tqdm

from ...asset import AssetManager
from .ism import InterShellMode as ISM
from .user_node import generate_sat2user, generate_user2sat, generate_is_user

import numpy as np
from . import MultiShellGraph as MSG
import pickle
import time
import copy
import networkx as nx
import scipy.io as sio
from tqdm import tqdm
from . import SPOnGrid as SPG
from . import ShortestPath as ShP

class StarlinkAdapter():
    
    def __init__(self, input_path, topo_file_template, file_volume, data_per_topo, ism:ISM, parallel=None):
        self.input_path = input_path
        self.input_topo_file_template = topo_file_template
        
        self.file_volume = file_volume
        self.data_per_topo = data_per_topo
        self.ism = ism
        
        self.parallel = parallel
        
    def adapt(self, output_path):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        os.makedirs(output_path)
        
        args = []
        for i in self.file_volume:
            file_path = os.path.join(self.input_path, self.input_topo_file_template.format(i))
            args.append((file_path, self.data_per_topo, i, self.ism, output_path))
        
        with mp.Pool(self.parallel) as pool:
            pool.starmap(StarlinkAdapter._adapt_topo_file, args)
        
        
    @staticmethod
    def _adapt_topo_file(file_path, data_num, file_idx, ism, output_path):
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

        starlink_dataset = []

        for k in tqdm(range(data_num)):
            
            data = pickle.load(file)

            G_interShell = data['InterShell_GrdRelay']
            ISL_interShell = data['InterShell_ISL']
            
            # # 1. Save pathform metadata
            # meta = StarlinkPathFormer.create_metadata(Offset5, GrdStationNum, data['InterShell_GrdRelay'], data['InterShell_ISL'])

            # AssetManager.save_pathform_metadata_(output_path, file_idx, k, meta)
            
            # 2. Cerate the topology
            match ism:
                case ISM.GRD_STATION:
                    E_inter = []
                    for SatIndex in range(len(G_interShell)):
                        if int(G_interShell[SatIndex]) >= 0:
                            E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset5)])
                            E_inter.append([int(G_interShell[SatIndex] + Offset5), SatIndex])  
                    graph_node_num = Offset5 * 2 + GrdStationNum

                    

                case ISM.ISL:
                    E_inter = []
                    for SatIndex in range(len(ISL_interShell[0])): # 2 to 1
                        if ISL_interShell[0][SatIndex] >= 0:
                            S2 = SatIndex + Offset2 
                            S1 = int(ISL_interShell[0][SatIndex])
                            E_inter.append([S2, S1])
                            E_inter.append([S1, S2])
                    
                    for SatIndex in range(len(ISL_interShell[1])): # 3 to 2
                        if ISL_interShell[1][SatIndex] >= 0:
                            S3 = int(SatIndex + Offset3)
                            S2 = int(ISL_interShell[1][SatIndex] + Offset2)
                            E_inter.append([S3, S2])
                            E_inter.append([S2, S3])
                    
                    for SatIndex in range(len(ISL_interShell[2])): # 4 to 3
                        if ISL_interShell[2][SatIndex] >= 0:
                            S4 = int(SatIndex + Offset4)
                            S3 = int(ISL_interShell[2][SatIndex] + Offset3)
                            E_inter.append([S4, S3])
                            E_inter.append([S3, S4])
                    graph_node_num = Offset5 * 2
                    
            sat2user = generate_sat2user(satellite_num, GrdStationNum, ism)
            
            E = E1 + E2 + E3 + E4 + E_inter
            # G = nx.DiGraph()
            # G.add_nodes_from(range(graph_node_num))
            # ## 1. Inter-satellite links
            # for e in E:
            #     G.add_edge(e[0], e[1], capacity=isl_cap)
            # ## 2. User-satellite links
            # for i in range(satellite_num):
            #     # Uplink
            #     G.add_edge(sat2user(i), i, capacity=uplink_cap)
            #     # Downlink
            #     G.add_edge(i, sat2user(i), capacity=downlink_cap)
            # ## 3. Inter ground station links
            # for i in range(GrdStationNum):
            #     for j in range(GrdStationNum):
            #         if i == j:
            #             continue
            #         G.add_edge(i + Offset5, j + Offset5, capacity=0)
            #         G.add_edge(j + Offset5, i + Offset5, capacity=0)
                
            # AssetManager.save_graph_(output_path, file_idx, k, G)

             # 3. Create traffic matrices

            tm_dict = {}
            path_dict = {}
            
            flowset = data['FlowSet']

            for flow in flowset:
                src = sat2user(flow[0])
                dst = sat2user(flow[1])
                d = flow[2]

                paths = SPG.SPOnGrid(flow[0],flow[1],
                        G_interShell, 
                        ISL_interShell, 
                        ism, 
                        5)

                while len(paths) < 5:
                    paths.append(paths[0])  
                
                path_dict[f'{src}, {dst}'] = [[src] + path + [dst] for path in paths]
                tm_dict[f'{src}, {dst}'] = d

            data_idx = k if file_idx == "A" else 5000 + k
                            
            # if k < train_num:
            #     # AssetManager.save_tm_train_separate_(output_path, k if file_idx == "A" else 5000 + k, 0, demand_matrix)
            #     starlink_dataset.append({'graph': E, 'tm': demand_matrix, 'meta': meta, 'data_idx': data_idx, 'train': True})
            # elif k < data_num:
            #     # AssetManager.save_tm_test_separate_(output_path, k - train_num if file_idx == "A" else 5000 + k - train_num, 0, demand_matrix)
            #     starlink_dataset.append({'graph': E, 'tm': demand_matrix, 'meta': meta, 'data_idx': data_idx, 'train': False})
            # else:
            #     break;

            starlink_dataset.append({'graph': E, 'tm': tm_dict, 'path': path_dict, 'data_idx': data_idx})

        file.close()

        AssetManager.save_dataset_(output_path, os.path.basename(file_path), starlink_dataset)
        

class StarlinkMixAdapter():

    def __init__(self, input_path, topo_file_template, data_per_topo, ism, parallel=None):
        self.input_path = input_path
        self.input_topo_file_template = topo_file_template
        
        self.data_per_topo = data_per_topo

        self.ism = ism
        
        self.parallel = parallel

    def adapt(self, output_path):

        data_num = self.data_per_topo

        ism = self.ism

        data_idx = 0

        INTENSITY = [25, 50, 75, 100]

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        os.makedirs(output_path)

        file_paths = [os.path.join(self.input_path, self.input_topo_file_template.format(intensity)) for intensity in INTENSITY]

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

        starlink_mixed_dataset = []

        for file_path in file_paths:

            file = open(file_path, 'rb')

            for k in tqdm(range(data_num)):
                
                data = pickle.load(file)

                G_interShell = data['InterShell_GrdRelay']
                ISL_interShell = data['InterShell_ISL']
                
                # # 1. Save pathform metadata
                # meta = StarlinkPathFormer.create_metadata(Offset5, GrdStationNum, data['InterShell_GrdRelay'], data['InterShell_ISL'])

                # AssetManager.save_pathform_metadata_(output_path, file_idx, k, meta)
                
                # 2. Cerate the topology
                match ism:
                    case ISM.GRD_STATION:
                        E_inter = []
                        for SatIndex in range(len(G_interShell)):
                            if int(G_interShell[SatIndex]) >= 0:
                                E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset5)])
                                E_inter.append([int(G_interShell[SatIndex] + Offset5), SatIndex])  
                        graph_node_num = Offset5 * 2 + GrdStationNum

                        

                    case ISM.ISL:
                        E_inter = []
                        for SatIndex in range(len(ISL_interShell[0])): # 2 to 1
                            if ISL_interShell[0][SatIndex] >= 0:
                                S2 = SatIndex + Offset2 
                                S1 = int(ISL_interShell[0][SatIndex])
                                E_inter.append([S2, S1])
                                E_inter.append([S1, S2])
                        
                        for SatIndex in range(len(ISL_interShell[1])): # 3 to 2
                            if ISL_interShell[1][SatIndex] >= 0:
                                S3 = int(SatIndex + Offset3)
                                S2 = int(ISL_interShell[1][SatIndex] + Offset2)
                                E_inter.append([S3, S2])
                                E_inter.append([S2, S3])
                        
                        for SatIndex in range(len(ISL_interShell[2])): # 4 to 3
                            if ISL_interShell[2][SatIndex] >= 0:
                                S4 = int(SatIndex + Offset4)
                                S3 = int(ISL_interShell[2][SatIndex] + Offset3)
                                E_inter.append([S4, S3])
                                E_inter.append([S3, S4])
                        graph_node_num = Offset5 * 2
                        
                sat2user = generate_sat2user(satellite_num, GrdStationNum, ism)
                
                E = E1 + E2 + E3 + E4 + E_inter
                # G = nx.DiGraph()
                # G.add_nodes_from(range(graph_node_num))
                # ## 1. Inter-satellite links
                # for e in E:
                #     G.add_edge(e[0], e[1], capacity=isl_cap)
                # ## 2. User-satellite links
                # for i in range(satellite_num):
                #     # Uplink
                #     G.add_edge(sat2user(i), i, capacity=uplink_cap)
                #     # Downlink
                #     G.add_edge(i, sat2user(i), capacity=downlink_cap)
                # ## 3. Inter ground station links
                # for i in range(GrdStationNum):
                #     for j in range(GrdStationNum):
                #         if i == j:
                #             continue
                #         G.add_edge(i + Offset5, j + Offset5, capacity=0)
                #         G.add_edge(j + Offset5, i + Offset5, capacity=0)
                    
                # AssetManager.save_graph_(output_path, file_idx, k, G)

                # 3. Create traffic matrices

                tm_dict = {}
                path_dict = {}
                
                flowset = data['FlowSet']

                for flow in flowset:
                    src = sat2user(flow[0])
                    dst = sat2user(flow[1])
                    d = flow[2]

                    paths = SPG.SPOnGrid(flow[0],flow[1],
                            G_interShell, 
                            ISL_interShell, 
                            ism, 
                            5)

                    while len(paths) < 5:
                        paths.append(paths[0])  
                    
                    path_dict[f'{src}, {dst}'] = [[src] + path + [dst] for path in paths]
                    tm_dict[f'{src}, {dst}'] = d

                data_idx += 1
                                
                # if k < train_num:
                #     # AssetManager.save_tm_train_separate_(output_path, k if file_idx == "A" else 5000 + k, 0, demand_matrix)
                #     starlink_dataset.append({'graph': E, 'tm': demand_matrix, 'meta': meta, 'data_idx': data_idx, 'train': True})
                # elif k < data_num:
                #     # AssetManager.save_tm_test_separate_(output_path, k - train_num if file_idx == "A" else 5000 + k - train_num, 0, demand_matrix)
                #     starlink_dataset.append({'graph': E, 'tm': demand_matrix, 'meta': meta, 'data_idx': data_idx, 'train': False})
                # else:
                #     break;

                starlink_mixed_dataset.append({'graph': E, 'tm': tm_dict, 'path': path_dict, 'data_idx': data_idx})

            file.close()

        AssetManager.save_dataset_(output_path, self.input_topo_file_template.format('Mixed') , starlink_mixed_dataset)


class IridiumAdapter():

    def __init__(self, input_path, topo_file, data_per_topo, parallel=None):
        self.input_path = input_path
        self.input_topo_file = topo_file
        
        self.data_per_topo = data_per_topo
        
        self.parallel = parallel
        
    def adapt(self, output_path):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        os.makedirs(output_path)

        file_path = os.path.join(self.input_path, self.input_topo_file)
        
        IridiumAdapter._adapt_topo_file(file_path, self.data_per_topo, output_path)

    @staticmethod
    def _adapt_topo_file(file_path, data_num, output_path):
        OrbitNum = 6
        SatNum   = 11
        TotalNum = OrbitNum*SatNum

        iridium_dataset = []

        file = open(file_path, 'rb')

        for data_idx in tqdm(range(data_num)):
            data = pickle.load(file)
            FlowSet = data['FlowSet']
            E = data['E']
            ISLCap = data['ISLCap'] 
            UpLinkCap = data['UpLinkCap'] 
            DownLinkCap = data['DownLinkCap']
            G = np.zeros((TotalNum,TotalNum)) + 999
            EMap = np.zeros((TotalNum,TotalNum)) + 999
            Eindex = 0
            for Edge in E:
                G[int(Edge[0])][int(Edge[1])] = 1
                EMap[Edge[0]][Edge[1]] = Eindex
                Eindex += 1

            tm_dict = {}
            path_dict = {}
            for flow in FlowSet:
                path_edge = ShP.k_Shortest(G, flow[0], flow[1], 5, 999, E, EMap)
                Path = []
                for p in path_edge:
                    path_node = []
                    for e in p['path']:
                        # print(p)
                        path_node.append(E[int(e)][0])
                    path_node.append(E[int(e)][1])
                    Path.append(path_node)
                while len(Path) < 5:
                    Path.append(Path[0])

                tm_dict[f'{flow[0]}, {flow[1]}'] = flow[2]
                path_dict[f'{flow[0]}, {flow[1]}'] = Path


            iridium_dataset.append({'graph': E, 'tm': tm_dict, 'path': path_dict, 'data_idx': data_idx})

        AssetManager.save_dataset_(output_path, os.path.basename(file_path), iridium_dataset)
                
                # for p in Path:
                #     if p[0] != flow[0] or p[-1] != flow[1]:
                #         print('Src Des Error!')
                #         break
                #     p_set = set(p)
                #     if len(p_set) != len(p):
                #         print('Loop!')
                #         break
                #     for node in range(len(p)-1):
                #         if G[int(p[node])][int(p[node+1])] != 1:
                #             print(['Invalid Edge!',int(p[node]), int(p[node+1])])
                #             break