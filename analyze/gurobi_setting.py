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
import random
import networkx as nx
import scipy.io as sio
import lib.data.starlink.SPOnGrid as SPG

# ========= Parameters to change ==========
DataSetSize = 100 #
fileName = open("/data/projects/11003765/sate/input/raw/starlink/DataSetForSaTE100/StarLink_DataSetForAgent100_5000_A.pkl", "rb")
InterConnectedMode = 'ISL'#'GrdStation' # 'ISL'
ISLCap = 50 * 4
UpLinkCap = 200 * 4
DownLinkCap = 200 * 4

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
for k in range(1):
    data = pickle.load(fileName)
    G_interShell = data['InterShell_GrdRelay']
    ISL_interShell = data['InterShell_ISL']
    FlowSet = data['FlowSet']
    print(['Index of Loaded Data:', k])
    
    # =================================================
    # --------------- Generate E and G ----------------
    # =================================================
    GrdRatio = 0.6
    ISLRatio = [72, 278, 420]
    
    
    if InterConnectedMode == 'GrdStation':
        # Generate E
        E_inter = []
        for SatIndex in range(len(G_interShell)):
            if int(G_interShell[SatIndex]) >= 0:
                if random.random() < GrdRatio:
                    E_inter.append([SatIndex, int(G_interShell[SatIndex] + Offset5)])
                    E_inter.append([int(G_interShell[SatIndex] + Offset5), SatIndex])    
        # G_interShell = np.zeros(4236) - 1
        
        if not np.array_equal(G_interShell,G_interShell_last):
            G_Trajectory.append(k)
            G_interShell_last = copy.deepcopy(G_interShell)
    else:
        temp = E1 + E2 + E3 + E4
        # Generate E

        E_inter = []
        print(f'2 to 1: from {len(temp)} to {len(temp) + len(ISL_interShell[0])}')
        print(f'failure rate: {ISLRatio[0]}/{len(ISL_interShell[0])}')

        for SatIndex in range(len(ISL_interShell[0])): # 2 to 1
            if ISL_interShell[0][SatIndex] >= 0:
                if random.random() < ISLRatio[0]/len(ISL_interShell[0]):
                    S2 = SatIndex + Offset2 
                    S1 = int(ISL_interShell[0][SatIndex])
                    E_inter.append([S2, S1])
                    E_inter.append([S1, S2])

        print(f'3 to 2: from {len(temp) + len(ISL_interShell[0])} to {len(temp) + len(ISL_interShell[0]) + len(ISL_interShell[1])}')
        print(f'failure rate: {ISLRatio[1]}/{len(ISL_interShell[1])}')

        for SatIndex in range(len(ISL_interShell[1])): # 3 to 2
            if ISL_interShell[1][SatIndex] >= 0:
                if random.random() < ISLRatio[1]/len(ISL_interShell[1]):
                    S3 = int(SatIndex + Offset3)
                    S2 = int(ISL_interShell[1][SatIndex] + Offset2)
                    E_inter.append([S3, S2])
                    E_inter.append([S2, S3])

        print(f'4 to 3: from {len(temp) + len(ISL_interShell[0]) + len(ISL_interShell[1])} to {len(temp) + len(ISL_interShell[0]) + len(ISL_interShell[1]) + len(ISL_interShell[2])}')
        print(f'failure rate: {ISLRatio[2]}/{len(ISL_interShell[2])}')
        
        for SatIndex in range(len(ISL_interShell[2])): # 4 to 3
            if ISL_interShell[2][SatIndex] >= 0:
                if random.random() < ISLRatio[2]/len(ISL_interShell[2]):
                    S4 = int(SatIndex + Offset4)
                    S3 = int(ISL_interShell[2][SatIndex] + Offset3)
                    E_inter.append([S4, S3])
                    E_inter.append([S3, S4])
        
        if ISL_interShell != ISL_interShell_last:
            ISL_Trajectory.append(k)
            ISL_interShell_last = copy.deepcopy(ISL_interShell)
    
    E = E1 + E2 + E3 + E4 + E_inter
     
    
    # print(len(E))
    # LinkFailure = 0.01
    # for edge in E:
    #     r_random = random.random()
    #     if r_random <= LinkFailure:
    #         E.remove(edge)
    # print(len(E))
    # print(['Number of Edge:', len(E)])
    # Generate G
    G = np.zeros((Offset5 + GrdStationNum, Offset5 + GrdStationNum)) + 999
    EMap = np.zeros((Offset5 + GrdStationNum, Offset5 + GrdStationNum)) + 999
    kk = 0
    for Edge in E:
        G[Edge[0]][Edge[1]] = 1
        EMap[Edge[0]][Edge[1]] = kk
        kk += 1
    # =================================================
    # sio.savemat('TrackChanges.mat',{'G_Trajectory':G_Trajectory,
    #                                 'ISL_Trajectory':ISL_Trajectory})
    # =============== Yen's Algorithm ===================
    
    # for flow in FlowSet:
    #     time_M0 = time.perf_counter()
    #     Path = SP.k_Shortest(G, flow[0], flow[1], 5, 999, E, EMap)
    #     print("Finish!")
    #     time_M1 = time.perf_counter()
    #     print([len(FlowSet),time_M1- time_M0])
    # ===================================================
    
    
    # ============== ECMP ====================
    # ISLCapVec = [ISLCap for x in range(len(E))]
    # probability = 0
    # for x in range(len(E)-len(E_inter),len(E)):
    #     if random.random() < probability:
    #         ISLCapVec[x] = 0
    # time_M0 = time.perf_counter()
    # Throughput, Ratio, MaxE = Baseline1.ECMP(E,FlowSet,G_interShell,ISL_interShell,InterConnectedMode,ISLCapVec,UpLinkCap,DownLinkCap)
    # time_M1 = time.perf_counter()
    # print([Throughput, Ratio, time_M1- time_M0])
    # Result[0].append(Throughput)
    # Result[1].append(Ratio)
    # Result[2].append(time_M1- time_M0)




    # # ============== Back Pressure Routing ============
    # # establish the Path set of each flow 
    # Path_Set = []
    # for flow in FlowSet: 
    #     Path = SPG.SPOnGrid(flow[0],flow[1],
    #                     G_interShell, 
    #                     ISL_interShell, 
    #                     InterConnectedMode, 
    #                     5) 
    #     Path_Set.append(Path)
    # # if k == 0:
    # #     Q_matrix = np.zeros((len(G), len(FlowSet)))
    # #     NonEmpty = [set() for node_a in range(len(G))]
    # ThroResult, Ratio, Q_matrix, NonEmpty = Baseline2.BackPressureRouting(FlowSet, 
    #                                                                       Path_Set, 
    #                                                                       G, 
    #                                                                       ISLCap, 
    #                                                                       2000)
    # print([sum(ThroResult), Ratio])



    
    # AccessNode = set()
    # for flow in FlowSet:
    #     AccessNode.add(flow[0])
    #     AccessNode.add(flow[1])
    # print([len(FlowSet),len(AccessNode)])

# Display result
# print(['ECMP Result:', [sum(Result[0])/len(Result[0]),
#        sum(Result[1])/len(Result[1]),
#        sum(Result[2])/len(Result[2])]])
            

    
