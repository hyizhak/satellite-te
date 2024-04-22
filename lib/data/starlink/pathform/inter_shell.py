# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:21:49 2024

@author: WH
"""
from . import intra_shell
from ..ism import InterShellMode as ISM

from . import intra_shell as LGP

def Layer(index):
    if index < 1584:
        return 0 
    elif index < 3168:
        return 1
    elif index < 3516:
        return 2
    else:
        return 3
        
def GotoNeibour(N,L1,L2,OrbitN,SatN,ISL_interShell,Offset,Already):
    S = N
    Path = [N + Offset[L1]]
    for index in range(L1-L2):
        S = ISL_interShell[L1 - index - 1][S]
        if S == -1:
            Already.append(N)
            # Two Neibour of N
            N1 = N//SatN*SatN + (N%SatN + 1)%SatN
            N2 = N//SatN*SatN + (N%SatN - 1)%SatN
            # print(['NEW',N1,N2])
            # K is the nearest node list in L1
            F1 = 0
            F2 = 0
            if N1 not in Already:
                Already.append(N1)
                K1 = GotoNeibour(N1,L1,L2,OrbitN,SatN,ISL_interShell,Offset,Already)
                F1 = 1
            if N2 not in Already:
                Already.append(N2)
                K2 = GotoNeibour(N2,L1,L2,OrbitN,SatN,ISL_interShell,Offset,Already)
                F2 = 1
            if F1 == 1 and F2 == 1:
                if len(K1) <= len(K2):
                    return [N + Offset[L1]] + K1
                else:
                    return [N + Offset[L1]] + K2   
            elif F2 == 0:
                return [N + Offset[L1]] + K1
            else:
                return [N + Offset[L1]] + K2
            
        else:
            Path.append(S + Offset[L1 - index -1])
    return Path

def SPBetweenGrd(g1, g2, G_interShell, Offset, pathN):
    g1_x = g1 - Offset
    g2_x = g2 - Offset
    G_MM = G_interShell[0:72*22]
    s1 = G_MM.index(g1_x)
    s2 = G_MM.index(g2_x)
    Path = LGP.k_shortest_path_loop_grid(s1, 
                                         s2, 
                                         0, 
                                         72, 
                                         22, 
                                         pathN)
    PathComplete = []
    for p in Path:
        PathComplete.append([g1]+p+[g2])

def SatOverGrdStation(grd, G_interShell):
    if grd in G_interShell:
        return G_interShell.index(grd)    
    else:
        return None

def SPOnGrid(src, des, G_interShell, ISL_interShell, mode, pathN):
    Offset = [0, 1584, 3168, 3516, 4236]
    OrbitN = [72, 72, 58, 36]
    SatN   = [22, 22, 6, 20]
    Lsrc = Layer(src)
    Ldes = Layer(des)
    # print(['Layers:',Lsrc,Ldes,src,des])
    if Lsrc == Ldes: # Intra-layer paths
        OffSD = Offset[Lsrc]
        # print([src, des, OffSD, OrbitN[Lsrc],SatN[Lsrc], pathN])
        # print(LGP.k_shortest_path_loop_grid(src, 
        #                                      des, 
        #                                      OffSD, 
        #                                      OrbitN[Lsrc], 
        #                                      SatN[Lsrc], 
        #                                      pathN))
        return LGP.k_shortest_path_loop_grid(src, 
                                             des, 
                                             OffSD, 
                                             OrbitN[Lsrc], 
                                             SatN[Lsrc], 
                                             pathN)
        
    
    else:
        if mode == 'GrdStation':
            LL = max(Lsrc,Ldes)
            MM = min(Lsrc,Ldes)
            # Find the node in LL that connects to a node in MM
            G_LL = G_interShell[Offset[LL]:Offset[LL+1]]
            G_MM = G_interShell[Offset[MM]:Offset[MM+1]]
            Inter = -1
            GS    = -1
            MinDis = 10000
            for index in range(len(G_LL)):
                if G_LL[index] != -1 and G_LL[index] in G_MM:
                     if Lsrc > Ldes:
                         if abs(src - Offset[Lsrc] - index) < MinDis:
                             MinDis = abs(src - Offset[Lsrc] - index)
                             Inter = G_MM.index(G_LL[index])
                             Inter2= index
                             GS    = G_LL[index]
                     else:
                         if abs(des - Offset[Ldes] - index) < MinDis:
                             MinDis = abs(des - Offset[Ldes] - index)
                             Inter = G_MM.index(G_LL[index])
                             Inter2= index
                             GS    = G_LL[index]
            OffS = Offset[Lsrc]
            OffD = Offset[Ldes]
            PathComplete = []
            if  Lsrc < Ldes:
                PathIntra_High = LGP.k_shortest_path_loop_grid(Inter2 + OffD, 
                                                               des, 
                                                               OffD, 
                                                               OrbitN[Ldes], 
                                                               SatN[Ldes], 
                                                               1)               
                # print([src - OffS, Inter, OffS, OrbitN[Lsrc], SatN[Lsrc], pathN])
                PathIntra = LGP.k_shortest_path_loop_grid(src, 
                                                          Inter + OffS, 
                                                          OffS, 
                                                          OrbitN[Lsrc], 
                                                          SatN[Lsrc], 
                                                          pathN)
                for p in PathIntra:
                    PathComplete.append(p+[GS + Offset[4]]+PathIntra_High[0])
                return PathComplete
            else:
                PathIntra_High = LGP.k_shortest_path_loop_grid(src, 
                                                               Inter2 + OffS, 
                                                               OffS, 
                                                               OrbitN[Lsrc], 
                                                               SatN[Lsrc], 
                                                               1)
                PathIntra = LGP.k_shortest_path_loop_grid(Inter + OffD, 
                                                          des, 
                                                          OffD, 
                                                          OrbitN[Ldes], 
                                                          SatN[Ldes], 
                                                          pathN)
                for p in PathIntra:
                    PathComplete.append(PathIntra_High[0] + [GS + Offset[4]] + p)
                return PathComplete
        else:
            if Lsrc > Ldes:
                N = src - Offset[Lsrc]
                PathCross = GotoNeibour(N,
                                        Lsrc,
                                        Ldes,
                                        OrbitN[Lsrc],
                                        SatN[Lsrc],
                                        ISL_interShell,
                                        Offset, [])
                # print(['PathCross',PathCross])
                SubSrc = PathCross.pop(-1)
                OffD = Offset[Ldes]              
                # print([SubSrc, des, OffD, OrbitN[Ldes], SatN[Ldes], pathN])
                PathIntra = LGP.k_shortest_path_loop_grid(SubSrc, 
                                                          des, 
                                                          OffD, 
                                                          OrbitN[Ldes], 
                                                          SatN[Ldes], 
                                                          pathN)
                PathComplete = []
                for p in PathIntra:
                    PathComplete.append(PathCross + p)
                # print(['PathCross',PathCross])
                return PathComplete
            else:
                N = des - Offset[Ldes]
                PathCross = GotoNeibour(N,
                                        Ldes,
                                        Lsrc,
                                        OrbitN[Ldes],
                                        SatN[Ldes],
                                        ISL_interShell,
                                        Offset, [])
                # print(['PathCross',PathCross])
                SubSrc = PathCross.pop(-1)
                PathCross.reverse()
                OffS = Offset[Lsrc]
                # print([src, SubSrc, OffS, OrbitN[Lsrc], SatN[Lsrc], pathN])
                PathIntra = LGP.k_shortest_path_loop_grid(src, 
                                                          SubSrc, 
                                                          OffS, 
                                                          OrbitN[Lsrc], 
                                                          SatN[Lsrc], 
                                                          pathN)
                # print('ZZ')
                PathComplete = []
                for p in PathIntra:
                    PathComplete.append(p + PathCross)
                return PathComplete
        
    