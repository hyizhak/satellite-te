import pickle
import numpy as np
import networkx as nx
import multiprocessing as mp
import os
from tqdm import tqdm

from .ism import InterShellMode as ISM
from .user_node import generate_sat2user, generate_user2sat, generate_is_user

import numpy as np
from . import MultiShellGraph as MSG
import pickle
import json
import networkx as nx
from tqdm import tqdm
from . import SPOnGrid as SPG
from . import SPOnGridReduced as SPGR
from . import ShortestPath as ShP

starlink_isl_cap = 200
starlink_uplink_cap = 800
starlink_downlink_cap = 800
iridium_isl_cap = 25
iridium_uplink_cap = 100
iridium_downlink_cap = 100

def add_edge(links, src, dst, capacity):
    links.append({
        "capacity": capacity,
        "source": src,
        "target": dst
    })

class StarlinkHARPAdapter():
    
    def __init__(self, input_path, topo_file_template, file_volume, data_per_topo, ism:ISM, parallel=None):
        self.input_path = input_path
        self.input_topo_file_template = topo_file_template
        
        self.file_volume = file_volume
        self.data_per_topo = data_per_topo
        self.ism = ism
        
        self.parallel = parallel
        
    def adapt(self, output_path,):
        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)
        
        # os.makedirs(output_path)

        assert os.path.exists(output_path)
        os.makedirs(f"{output_path}/manifest", exist_ok=True)
        os.makedirs(f"{output_path}/topologies/starlink_4000", exist_ok=True)
        os.makedirs(f"{output_path}/topologies/paths_dict", exist_ok=True)
        os.makedirs(f"{output_path}/pairs/starlink_4000", exist_ok=True)
        os.makedirs(f"{output_path}/traffic_matrices/starlink_4000", exist_ok=True)
        
        args = []
        for i in self.file_volume:
            file_path = os.path.join(self.input_path, self.input_topo_file_template.format(i))
            args.append((file_path, self.data_per_topo, i, self.ism, output_path))
        
        with mp.Pool(self.parallel) as pool:
            pool.starmap(StarlinkHARPAdapter._adapt_topo_file, args)
        
        
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

            links = []
            for e in E:
                add_edge(links, e[0], e[1], capacity=starlink_isl_cap)
            ## 2. User-satellite links
            for i in range(satellite_num):
                # Uplink
                add_edge(links, sat2user(i), i, capacity=starlink_uplink_cap)
                # Downlink
                add_edge(links, i, sat2user(i), capacity=starlink_downlink_cap)
            ## 3. Inter ground station links
            for i in range(GrdStationNum):
                for j in range(GrdStationNum):
                    if i == j:
                        continue
                    add_edge(links, i + Offset5, j + Offset5, capacity=0)
                    add_edge(links, j + Offset5, i + Offset5, capacity=0)

            topo_dict = {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": i} for i in range(graph_node_num)],
                "links": links
            }

            with open(f"{output_path}/topologies/starlink_4000/t{k}.json", "w") as f:
                json.dump(topo_dict, f)

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
                
                path_dict[(src, dst)] = [[src] + path + [dst] for path in paths]
                tm_dict[(src, dst)] = tm_dict.get((src, dst), 0) + d
                
            # 1) Sort the pairs to fix an ordering
            pairs = sorted(path_dict.keys())   # e.g. [(0,1), (0,2), (1,2), …]

            # 2) Build the NumPy arrays
            #    - pairs_array: shape (num_pairs,2)
            #    - traffic_array: shape (num_pairs,1)
            pairs_array = np.array(pairs, dtype=np.int32)
            traffic_array = np.array([[tm_dict[p]] for p in pairs], dtype=np.float32)

            with open(f"{output_path}/pairs/starlink_4000/t{k}.pkl", "wb") as f:
                pickle.dump(list(pairs_array), f)
            with open(f"{output_path}/traffic_matrices/starlink_4000/t{k}.pkl", "wb") as f:
                pickle.dump(list(traffic_array), f)

            with open(f"{output_path}/manifest/starlink_4000_manifest.txt", "a") as f:
                f.write(f"t{k}.json,t{k}.pkl,t{k}.pkl\n") 

            # 3) Paths
            edge_paths = {}
            for pair, paths in path_dict.items():
                ep = []
                for node_seq in paths:
                    # zip adjacent nodes into edges
                    edges = list(zip(node_seq[:-1], node_seq[1:]))
                    ep.append(edges)
                edge_paths[pair] = ep

            with open(f"{output_path}/topologies/paths_dict/starlink_4000_5_paths_dict_cluster_{k}.pkl", "wb") as f:
                pickle.dump(edge_paths, f)

        file.close()

class StarlinkReducedHARPAdapter():

    def __init__(self, input_path, topo_file_template, data_per_topo, ism, reduced, parallel=None):
        self.input_path = input_path
        self.input_topo_file_template = topo_file_template
        
        self.data_per_topo = data_per_topo

        self.ism = ism

        self.reduced = reduced
        
        self.parallel = parallel

    def adapt(self, output_path):

        data_num = self.data_per_topo

        ism = self.ism

        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)
        
        # os.makedirs(output_path)

        assert os.path.exists(output_path)

        match self.reduced:
            case 18:
                size = 176
            case 8:
                size = 500
            case 6:
                size = 528
            case 2:
                size = 1500

        os.makedirs(f"{output_path}/manifest", exist_ok=True)
        os.makedirs(f"{output_path}/topologies/starlink_{size}", exist_ok=True)
        os.makedirs(f"{output_path}/topologies/paths_dict", exist_ok=True)
        os.makedirs(f"{output_path}/pairs/starlink_{size}", exist_ok=True)
        os.makedirs(f"{output_path}/traffic_matrices/starlink_{size}", exist_ok=True)

        file_path = os.path.join(self.input_path, self.input_topo_file_template)

        # ========= Orbit Shell Parameters =========
        OrbitNum1 = round(72/self.reduced)
        SatNum1   = 22

        OrbitNum2 = round(72/self.reduced)
        SatNum2   = 22

        GrdStationNum = 222
        
        # =========== Generate Intra-Shell Static Graph =========
        LatMat1 = [0 for i in range(OrbitNum1 * SatNum1)]
        LatMat2 = [0 for i in range(OrbitNum2 * SatNum2)]
        LatLimit = 90

        Offset1 = 0
        Offset2 = OrbitNum1 * SatNum1
        Offset3 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2
        
        satellite_num = Offset3

        G1, EMap1, E1 = MSG.Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, Offset1, LatLimit)
        G2, EMap2, E2 = MSG.Inter_Shell_Graph(OrbitNum2, SatNum2, LatMat2, Offset2, LatLimit)

        with open(file_path, 'rb') as file:

            for k in tqdm(range(data_num)):
                
                data = pickle.load(file)
                
                G_interShell = data['InterShell_GrdRelay']
                ISL_interShell = data['InterShell_ISL']

                match ism:
                    case ISM.GRD_STATION:
                        E_inter = []
                        for SatIndex in range(len(G_interShell)):
                            if int(G_interShell[SatIndex]) >= 0:
                                E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset3)])
                                E_inter.append([int(G_interShell[SatIndex] + Offset3), SatIndex])   

                        graph_node_num = Offset3 * 2 + GrdStationNum

                    case ISM.ISL:
                        E_inter = []
                        for SatIndex in range(len(ISL_interShell[0])): # 2 to 1
                            if ISL_interShell[0][SatIndex] >= 0:
                                S2 = SatIndex + Offset2 
                                S1 = int(ISL_interShell[0][SatIndex])
                                E_inter.append([S2, S1])
                                E_inter.append([S1, S2])

                        graph_node_num = Offset3 * 2
                        
                sat2user = generate_sat2user(satellite_num, GrdStationNum, ism)
                
                E = E1 + E2 + E_inter

                links = []
                for e in E:
                    add_edge(links, e[0], e[1], capacity=starlink_isl_cap)
                ## 2. User-satellite links
                for i in range(satellite_num):
                    # Uplink
                    add_edge(links, sat2user(i), i, capacity=starlink_uplink_cap)
                    # Downlink
                    add_edge(links, i, sat2user(i), capacity=starlink_downlink_cap)
                ## 3. Inter ground station links
                for i in range(GrdStationNum):
                    for j in range(GrdStationNum):
                        if i == j:
                            continue
                        add_edge(links, i + Offset3, j + Offset3, capacity=0)
                        add_edge(links, j + Offset3, i + Offset3, capacity=0)

                topo_dict = {
                    "directed": True,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [{"id": i} for i in range(graph_node_num)],
                    "links": links
                }

                with open(f"{output_path}/topologies/starlink_{size}/t{k}.json", "w") as f:
                    json.dump(topo_dict, f)

                tm_dict = {}
                path_dict = {}
                
                flowset = data['FlowSet']

                for flow in flowset:
                    src = sat2user(flow[0])
                    dst = sat2user(flow[1])
                    d = flow[2]

                    paths = SPGR.SPOnGrid(flow[0],flow[1],
                            G_interShell, 
                            ISL_interShell, 
                            ism, 
                            5,
                            self.reduced)

                    while len(paths) < 5:
                        paths.append(paths[0])  
                    
                    path_dict[(src, dst)] = [[src] + path + [dst] for path in paths]
                    tm_dict[(src, dst)] = tm_dict.get((src, dst), 0) + d

                # 1) Sort the pairs to fix an ordering
                pairs = sorted(path_dict.keys())   # e.g. [(0,1), (0,2), (1,2), …]

                # 2) Build the NumPy arrays
                #    - pairs_array: shape (num_pairs,2)
                #    - traffic_array: shape (num_pairs,1)
                pairs_array = np.array(pairs, dtype=np.int32)
                traffic_array = np.array([[tm_dict[p]] for p in pairs], dtype=np.float32)

                with open(f"{output_path}/pairs/starlink_{size}/t{k}.pkl", "wb") as f:
                    pickle.dump(list(pairs_array), f)
                with open(f"{output_path}/traffic_matrices/starlink_{size}/t{k}.pkl", "wb") as f:
                    pickle.dump(list(traffic_array), f)

                with open(f"{output_path}/manifest/starlink_{size}_manifest.txt", "a") as f:
                    f.write(f"t{k}.json,t{k}.pkl,t{k}.pkl\n") 

                # 3) Paths
                edge_paths = {}
                for pair, paths in path_dict.items():
                    ep = []
                    for node_seq in paths:
                        # zip adjacent nodes into edges
                        edges = list(zip(node_seq[:-1], node_seq[1:]))
                        ep.append(edges)
                    edge_paths[pair] = ep

                with open(f"{output_path}/topologies/paths_dict/starlink_{size}_5_paths_dict_cluster_{k}.pkl", "wb") as f:
                    pickle.dump(edge_paths, f)


class IridiumHARPAdapter():

    def __init__(self, input_path, topo_file, data_per_topo, parallel=None):
        self.input_path = input_path
        self.input_topo_file = topo_file
        
        self.data_per_topo = data_per_topo
        
        self.parallel = parallel
        
    def adapt(self, output_path):
        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)
        
        # os.makedirs(output_path)

        os.makedirs(f"{output_path}/manifest", exist_ok=True)
        os.makedirs(f"{output_path}/topologies/iridium", exist_ok=True)
        os.makedirs(f"{output_path}/topologies/paths_dict", exist_ok=True)
        os.makedirs(f"{output_path}/pairs/iridium", exist_ok=True)
        os.makedirs(f"{output_path}/traffic_matrices/iridium", exist_ok=True)

        file_path = os.path.join(self.input_path, self.input_topo_file)
        
        IridiumHARPAdapter._adapt_topo_file(file_path, self.data_per_topo, output_path)

    @staticmethod
    def _adapt_topo_file(file_path, data_num, output_path):
        OrbitNum = 6
        SatNum   = 11
        TotalNum = OrbitNum*SatNum

        file = open(file_path, 'rb')

        for data_idx in tqdm(range(data_num)):
            data = pickle.load(file)
            FlowSet = data['FlowSet']
            E = data['E']

            links = []
            for e in E:
                add_edge(links, e[0], e[1], capacity=iridium_isl_cap)
            ## 2. User-satellite links
            for i in range(TotalNum):
                # Uplink
                add_edge(links, i+TotalNum, i, capacity=iridium_uplink_cap)
                # Downlink
                add_edge(links, i, i+TotalNum, capacity=iridium_downlink_cap)

            topo_dict = {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": i} for i in range(TotalNum*2)],
                "links": links
            }

            with open(f"{output_path}/topologies/iridium/t{data_idx}.json", "w") as f:
                json.dump(topo_dict, f)

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
                src = flow[0] + TotalNum
                dst = flow[1] + TotalNum
                d = flow[2]
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

                tm_dict[(src, dst)] = tm_dict.get((src, dst), 0) + d
                path_dict[(src, dst)] = [[src] + path + [dst] for path in Path]

            # 1) Sort the pairs to fix an ordering
            pairs = sorted(path_dict.keys())   # e.g. [(0,1), (0,2), (1,2), …]

            # 2) Build the NumPy arrays
            #    - pairs_array: shape (num_pairs,2)
            #    - traffic_array: shape (num_pairs,1)
            pairs_array = np.array(pairs, dtype=np.int32)
            traffic_array = np.array([[tm_dict[p]] for p in pairs], dtype=np.float32)

            with open(f"{output_path}/pairs/iridium/t{data_idx}.pkl", "wb") as f:
                pickle.dump(list(pairs_array), f)
            with open(f"{output_path}/traffic_matrices/iridium/t{data_idx}.pkl", "wb") as f:
                pickle.dump(list(traffic_array), f)

            with open(f"{output_path}/manifest/iridium_manifest.txt", "a") as f:
                f.write(f"t{data_idx}.json,t{data_idx}.pkl,t{data_idx}.pkl\n")

            # 3) Paths
            edge_paths = {}
            for pair, paths in path_dict.items():
                ep = []
                for node_seq in paths:
                    # zip adjacent nodes into edges
                    edges = list(zip(node_seq[:-1], node_seq[1:]))
                    ep.append(edges)
                edge_paths[pair] = ep

            with open(f"{output_path}/topologies/paths_dict/iridium_5_paths_dict_cluster_{data_idx}.pkl", "wb") as f:
                pickle.dump(edge_paths, f)
        
        file.close()
