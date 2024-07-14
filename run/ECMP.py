# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:26 2024

@author: WHH
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import lib.data.starlink.MultiShellGraph as MSG
import pickle
import time
import copy
import networkx as nx
import scipy.io as sio
import lib.data.starlink.SPOnGrid as SPG
import lib.data.starlink.ECMP as Baseline1

def ECMP(intensity, mode) :
    # ========= Parameters to change ==========
    DataSetSize = 100 #
    fileName = open(f"/data/projects/11003765/sate/input/starlink/DataSetForSaTE{intensity}/StarLink_DataSetForAgent{intensity}_5000_A.pkl", "rb")
    InterConnectedMode = mode #'GrdStation' # 'ISL'
    ISLCap = 50*4
    UpLinkCap = 200*4
    DownLinkCap = 200*4

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
    Result = [[],[],[]]
    # =========== Load DataSet ==============
    for k in range(DataSetSize):
        data = pickle.load(fileName)
        G_interShell = data['InterShell_GrdRelay']
        ISL_interShell = data['InterShell_ISL']
        FlowSet = data['FlowSet']
        # print(['Index of Loaded Data:', k])
        
        # =================================================
        # --------------- Generate E and G ----------------
        # =================================================
        if InterConnectedMode == 'GrdStation':
            # Generate E
            E_inter = []
            for SatIndex in range(len(G_interShell)):
                if int(G_interShell[SatIndex]) >= 0:
                    E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset5)])
                    E_inter.append([int(G_interShell[SatIndex] + Offset5), SatIndex])    
            # G_interShell = np.zeros(4236) - 1
            
            if not np.array_equal(G_interShell,G_interShell_last):
                G_Trajectory.append(k)
                G_interShell_last = copy.deepcopy(G_interShell)
        else:
            # Generate E
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
        # sio.savemat('TrackChanges.mat',{'G_Trajectory':G_Trajectory,
        #                                 'ISL_Trajectory':ISL_Trajectory})
        # ======================== Routing ================
        
        # E_load = [0 for x in range(len(E))]
        # for flow in FlowSet :
        #     Path = SPG.SPOnGrid(flow[0],flow[1],
        #                     G_interShell, 
        #                     ISL_interShell, 
        #                     InterConnectedMode, 
        #                     5) 
        #     for p in Path:
        #         for node in range(len(p)-1):
        #             x = E.index([int(p[node]),int(p[node+1])])
        #             E_load[x] = E_load[x] + flow[2]/5
        # print(sum(min(200,x) for x in E_load) / sum(x > 0 for x in E_load))
        # print(sum(x for x in E_load) / sum(x > 0 for x in E_load))   
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
        # for grd in range(222):
        #     print(SPG.SatOverGrdStation(grd,G_interShell))
        # ============== ECMP ====================
        time_M0 = time.perf_counter()
        Throughput, Ratio = Baseline1.ECMP(E,FlowSet,G_interShell,ISL_interShell,InterConnectedMode,ISLCap,UpLinkCap,DownLinkCap)
        time_M1 = time.perf_counter()
        # print([Throughput, Ratio, time_M1- time_M0])
        Result[0].append(Throughput)
        Result[1].append(Ratio)
        Result[2].append(time_M1- time_M0)

    print(f"ECMP Results for Intensity: {intensity} and mode {mode}")
    print([sum(Result[0])/len(Result[0]),
        sum(Result[1])/len(Result[1]),
        sum(Result[2])/len(Result[2])])
                
        # ================ Isomorphism ==================
        # if k == 0:
        #     Purified_Topology = []  
        # IsANewTopology = 1
        # for Paras in Purified_Topology:  
        #     start = time.perf_counter()
        #     TT = nx.Graph()
        #     TT.add_nodes_from([i for i in range(Offset5 + GrdStationNum)])
        #     TT.add_edges_from(E)
        #     T0 = nx.Graph()
        #     T0.add_nodes_from([i for i in range(Offset5 + GrdStationNum)])
        #     T0.add_edges_from(E1+E2+E3+E4+Paras)
        #     GM = nx.isomorphism.GraphMatcher(T0, TT)
        #     # print(GM.is_isomorphic())
        #     if nx.is_isomorphic(T0, TT) is True:            
        #         # mapping = GM.mapping
        #         IsANewTopology = 0
        #         break
        #     end = time.perf_counter() 
        #     print(end-start)
        # if IsANewTopology == 1:
        #     Purified_Topology.append(E_inter)
        # print([len(G_Trajectory),len(ISL_Trajectory),len(Purified_Topology)])   
        

if __name__ == '__main__':

    intensity = sys.argv[1]
    mode = sys.argv[2]
    ECMP(intensity, mode)