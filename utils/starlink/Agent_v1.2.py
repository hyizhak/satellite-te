import numpy as np
import MultiShellGraph as MSG
import pickle
import time
import copy
import networkx as nx
# ========= Parameters to change ==========
DataSetSize = 6800 #
fileName = open("StarLink_DataSet_50_6800.pkl", "rb")
InterConnectedMode = 'GrdStation' # 'ISL'
ISLCap = 50
UpLinkCap = 200
DownLinkCap = 200

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

G1, EMap1, E1 = MSG.Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, Offset1, LatLimit)
G2, EMap2, E2 = MSG.Inter_Shell_Graph(OrbitNum2, SatNum2, LatMat2, Offset2, LatLimit)
G3, EMap3, E3 = MSG.Inter_Shell_Graph(OrbitNum3, SatNum3, LatMat3, Offset3, LatLimit)
G4, EMap4, E4 = MSG.Inter_Shell_Graph(OrbitNum4, SatNum4, LatMat4, Offset4, LatLimit)

G_Trajectory = []
ISL_Trajectory = []
G_interShell_last = []
ISL_interShell_last = []

# =========== Load DataSet ==============
for k in range(DataSetSize):
    data = pickle.load(fileName)
    G_interShell = data['InterShell_GrdRelay']
    ISL_interShell = data['InterShell_ISL']
    FlowSet = data['FlowSet']
    print(['Index of Loaded Data:', k])
    
    # =================================================
    # --------------- Generate E and G ----------------
    # =================================================
    if InterConnectedMode == 'GrdStation':
        # Generate E
        E_inter = []
        for SatIndex in range(len(G_interShell)):
            E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset5)])
            E_inter.append([int(G_interShell[SatIndex] + Offset5), SatIndex])    
        G_interShell = np.zeros(4236) - 1
        
        if not np.array_equal(G_interShell,G_interShell_last):
            G_Trajectory.append(k)
            G_interShell_last = copy.deepcopy(G_interShell)
    else:
        # Generate E
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
        
        if ISL_interShell != ISL_interShell_last:
            ISL_Trajectory.append(k)
            ISL_interShell_last = copy.deepcopy(ISL_interShell)
    
    E = E1 + E2 + E3 + E4 + E_inter
    # print(['Number of Edge:', len(E)])
    # Generate G
    G = np.zeros((Offset5 + GrdStationNum, Offset5 + GrdStationNum)) + 999
    for Edge in E:
        G[Edge[0]][Edge[1]] = 1
    # =================================================

    
    if k == 0:
        Purified_Topology = []  
    IsANewTopology = 1
    for Paras in Purified_Topology:  
        start = time.perf_counter()
        TT = nx.Graph()
        TT.add_nodes_from([i for i in range(Offset5 + GrdStationNum)])
        TT.add_edges_from(E)
        T0 = nx.Graph()
        T0.add_nodes_from([i for i in range(Offset5 + GrdStationNum)])
        T0.add_edges_from(E1+E2+E3+E4+Paras)
        GM = nx.isomorphism.GraphMatcher(T0, TT)
        # print(GM.is_isomorphic())
        if nx.is_isomorphic(T0, TT) is True:            
            # mapping = GM.mapping
            IsANewTopology = 0
            break
        end = time.perf_counter() 
        print(end-start)
    if IsANewTopology == 1:
        Purified_Topology.append(E_inter)
    print([len(G_Trajectory),len(ISL_Trajectory),len(Purified_Topology)])   
    