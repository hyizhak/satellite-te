# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:27:49 2024

@author: admin
"""
from . import SPOnGrid as SPG
from .ism import InterShellMode as ISM

def ECMP(E,FlowSet,G_interShell,ISL_interShell,InterConnectedMode,ISLCap,UpLinkCap,DownLinkCap):
    Load_E = [0 for x in range(len(E))]
    Load_Up = [0 for x in range(4236)]
    Load_Down = [0 for x in range(4236)]
    Throughput = 0
    Demand = 0
    ism = ISM.ISL if InterConnectedMode == 'ISL' else ISM.GRD_STATION
    for flow in FlowSet:
        Path = SPG.SPOnGrid(flow[0],flow[1],
                        G_interShell, 
                        ISL_interShell, 
                        ism, 
                        5) 
        Demand += flow[2]
        b_Flow = min(UpLinkCap - Load_Up[flow[0]], DownLinkCap - Load_Down[flow[1]])
        
        for p in Path:
            
            for node in range(len(p)-1):
                x = E.index([int(p[node]),int(p[node+1])])
                b_Flow = min(b_Flow, flow[2]/len(Path), ISLCap - Load_E[x])
            
            for node in range(len(p)-1):
                x = E.index([int(p[node]),int(p[node+1])])
                Load_E[x] += b_Flow
            
            Load_Up[flow[0]] += b_Flow
            Load_Down[flow[1]] += b_Flow        
        
            Throughput += b_Flow
    return Throughput, Throughput/Demand
                