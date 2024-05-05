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
import lib.data.starlink.SPOnGridReduced as SPG
import lib.data.starlink.ECMPReduced as Baseline1

def ECMP(reduced, mode):
    # ========= Parameters to change ==========
    DataSetSize = 100 #
    size = 500 if int(reduced) == 8 else 1500
    fileName = open(f"/home/azureuser/cloudfiles/code/Users/e1310988/satellite-te/raw_data/starlink_{size}/StarLink_DataSetForAgent100_5000_Size{size}.pkl", "rb")
    InterConnectedMode = mode #'GrdStation' # 'ISL'
    ISLCap = 50 * 4
    UpLinkCap = 200 * 4
    DownLinkCap = 200 * 4
    ReduceFactor = int(reduced)
    # ========= Orbit Shell Parameters =========
    OrbitNum1 = round(72/ReduceFactor)
    SatNum1   = 22

    OrbitNum2 = round(72/ReduceFactor)
    SatNum2   = 22

    GrdStationNum = 222

    # =========== Generate Intra-Shell Static Graph =========
    LatMat1 = [0 for i in range(OrbitNum1 * SatNum1)]
    LatMat2 = [0 for i in range(OrbitNum2 * SatNum2)]
    LatLimit = 90

    Offset1 = 0
    Offset2 = OrbitNum1 * SatNum1
    Offset3 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2
    # Offset4 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2 + OrbitNum3 * SatNum3
    # Offset5 = OrbitNum1 * SatNum1 + OrbitNum2 * SatNum2 + OrbitNum3 * SatNum3 + OrbitNum4 * SatNum4

    G1, EMap1, E1 = MSG.Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, Offset1, LatLimit)
    G2, EMap2, E2 = MSG.Inter_Shell_Graph(OrbitNum2, SatNum2, LatMat2, Offset2, LatLimit)
    # G3, EMap3, E3 = MSG.Inter_Shell_Graph(OrbitNum3, SatNum3, LatMat3, Offset3, LatLimit)
    # G4, EMap4, E4 = MSG.Inter_Shell_Graph(OrbitNum4, SatNum4, LatMat4, Offset4, LatLimit)

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

        # =================================================
        # --------------- Generate E and G ----------------
        # =================================================
        if InterConnectedMode == 'GrdStation':
            # Generate E
            E_inter = []
            for SatIndex in range(len(G_interShell)):
                if int(G_interShell[SatIndex]) >= 0:
                    E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset3)])
                    E_inter.append([int(G_interShell[SatIndex] + Offset3), SatIndex])    
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
        
        E = E1 + E2 + E_inter
        # print(['Number of Edge:', len(E)])
        # Generate G
        G = np.zeros((Offset3 + GrdStationNum, Offset3 + GrdStationNum)) + 999
        for Edge in E:
            G[Edge[0]][Edge[1]] = 1
        # =================================================
        # sio.savemat('TrackChanges.mat',{'G_Trajectory':G_Trajectory,
        #                                 'ISL_Trajectory':ISL_Trajectory})
        # ======================== Routing ================
        
        for flow in FlowSet:
            Path = SPG.SPOnGrid(flow[0],flow[1],
                            G_interShell, 
                            ISL_interShell, 
                            InterConnectedMode, 
                            5,
                            ReduceFactor) 
            
            # for p in Path:
            #     if p[0] != flow[0] or p[-1] != flow[1]:
            #         print('Src Des Error!')
            #         break
            #     p_set = set(p)
            #     if len(p_set) != len(p):
            #         print('Loop!')
            #         break
            #     for node in range(len(p)-2):
            #         if G[int(p[node])][int(p[node+1])] != 1:
            #             print(['Invalid Edge!',int(p[node]), int(p[node+1])])
            #             break
        # ================== ECMP ============================
        time_M0 = time.perf_counter()
        Throughput, Ratio = Baseline1.ECMP(E,FlowSet,G_interShell,ISL_interShell,
                                        InterConnectedMode,ISLCap,UpLinkCap,
                                        DownLinkCap,ReduceFactor)
        time_M1 = time.perf_counter()
        # print([Throughput, Ratio, time_M1- time_M0])
        Result[0].append(Throughput)
        Result[1].append(Ratio)
        Result[2].append(time_M1- time_M0)

    print(f"ECMP Results for size: {size} and mode {mode}")
    print([sum(Result[0])/len(Result[0]),
        sum(Result[1])/len(Result[1]),
        sum(Result[2])/len(Result[2])])

if __name__ == '__main__':

    reduced = sys.argv[1]
    mode = sys.argv[2]
    ECMP(reduced, mode)
