import numpy as np
from astropy import units as u
import math

def Inter_Shell_Graph(OrbitNum1, SatNum1, LatMat1, OffSet, LatLimit):
    G = np.zeros((OrbitNum1 * SatNum1,
                  OrbitNum1 * SatNum1),dtype = int) + 999 # Adjecent Matrix
    EMap = np.zeros((OrbitNum1 * SatNum1, 
                     OrbitNum1 * SatNum1),dtype = int) + 999
    E = []
    edge_index = 0
    for orb in range(0,OrbitNum1):
        for sat in range(0,SatNum1):
            sat_index = orb * SatNum1 + sat 
            #----------------Inter-Orbit------------------------
            if abs(LatMat1[sat_index]) < LatLimit:
                # if orb == 0:
                #     neibor = (orb+1)*SatNum1 + sat # 0和5不连接，只和1连接
                #     if abs(LatMat1[neibor]) < LatLimit:
                #         G[sat_index,neibor] = 1
                #         EMap[sat_index,neibor] = edge_index
                #         E.append([int(sat_index + OffSet),int(neibor + OffSet)])
                #         edge_index += 1
                # elif orb == OrbitNum1 - 1: #5只和4连接
                #     neibor = (orb-1)*SatNum1 + sat    
                #     if abs(LatMat1[neibor]) < LatLimit:
                #         G[sat_index,neibor] = 1
                #         EMap[sat_index,neibor] = edge_index
                #         E.append([int(sat_index + OffSet),int(neibor + OffSet)])
                #         edge_index += 1
                # else:
                neibor1 = ((orb-1)% OrbitNum1) *SatNum1 + sat
                if abs(LatMat1[neibor1]) < LatLimit:                       
                    G[sat_index,neibor1] = 1                
                    EMap[sat_index,neibor1] = edge_index                
                    E.append([int(sat_index + OffSet),int(neibor1 + OffSet)])
                    edge_index += 1
                neibor2 = ((orb+1)% OrbitNum1) *SatNum1 + sat
                if abs(LatMat1[neibor2]) < LatLimit: 
                    G[sat_index,neibor2] = 1
                    EMap[sat_index,neibor2] = edge_index 
                    E.append([int(sat_index + OffSet),int(neibor2 + OffSet)])
                    edge_index += 1
            #-------------Intra-Orbit-------------------------        
            if sat == 0:
                sat_neibor3 = SatNum1 - 1
                sat_neibor4 = sat + 1
            elif sat == SatNum1 - 1:
                sat_neibor3 = sat - 1
                sat_neibor4 = 0
            else:
                sat_neibor3 = sat - 1
                sat_neibor4 = sat + 1
            neibor3 = (orb*SatNum1 + sat_neibor3)                     
            neibor4 = (orb*SatNum1 + sat_neibor4) 
            G[sat_index,neibor3] = 1
            G[sat_index,neibor4] = 1
            EMap[sat_index,neibor3] = edge_index
            EMap[sat_index,neibor4] = edge_index + 1
            E.append([int(sat_index + OffSet),int(neibor3 + OffSet)])
            E.append([int(sat_index + OffSet),int(neibor4 + OffSet)])
            edge_index += 2
    return G, EMap, E        

def Inter_Shell_Position(OrbitNum1, SatNum1, SL_Orb):
    LatMat1 = []
    LonMat1 = [] 
    xyz     = []      
    for orb_index in range(OrbitNum1):
        for sat_index in range(SatNum1):
            SL_Orb_Item1 = SL_Orb[orb_index].propagate(SL_Orb[orb_index].period/SatNum1*sat_index << u.s)
            [x,y,z] = SL_Orb_Item1.r.value
            xyz.append([x,y,z])
            Lat = math.atan(z/(x**2 + y**2)**0.5)*180/math.pi
            Lon = math.atan(y/x)*180/math.pi
            if x > 0:
                Lon = math.atan(y/x)*180/math.pi
            elif y > 0:
                Lon = 180 + math.atan(y/x)*180/math.pi
            else:
                Lon = -180 + math.atan(y/x)*180/math.pi
            LatMat1.append(Lat)
            LonMat1.append(Lon)
    return LatMat1, LonMat1, xyz