# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:26:26 2023

@author: WHH

Reference: https://en.wikipedia.org/wiki/Yen%27s_algorithm
"""
import numpy as np
import copy
from operator import itemgetter
def Shortest_Path(G, EMap, src, Max):
    n,col = G.shape # n = col = 节点数
    D = np.zeros(n) #最短路径长度
    p = [[] for i in range(n)]# np.zeros((row,EMap.max()+1),dtype=bool) #最短路径包含的边
    # print(len(self.E))
    final = np.zeros(n) #是否找到最短路径
    for v in range(0,n):
        final[v] = 0
        D[v] = G[src][v] #初始最短路径直接从邻接矩阵获得
        if D[v] < Max: #非临界
            edge = EMap[src,v]
            p[v].append(int(edge))
    D[src] = 0
    final[src] = 1
    for i in range(0,n):
        Min = Max
        for w in range(0,n):
            if final[w] == 0:
                if D[w] < Min:
                    v = w
                    Min = D[w]
        final[v] = 1
        for w in range(0,n):
            if((final[w]==0) and (Min + G[v][w] < D[w])):
                D[w] = Min + G[v][w]
                p[w] = copy.deepcopy(p[v])
                edge = EMap[v,w]
                p[w].append(int(edge))
    return D,p

def k_Shortest(G, src, des, max_k, Max, E, EMap):
    G0 = np.copy(G)
    D, p = Shortest_Path(G0, EMap, src, Max)
    A = [{'cost':D[des],'path':p[des]}]
    
    # path = p[des]
    # SS = src
    # for edge in path:
    #     edge_src = E[edge][0]
    #     if edge_src != SS:
    #         print('src error!')
    #     SS = E[edge][1]
    # if SS!= des:
    #     print(['des error!',SS,des])
        
    B = []   
    for k in range(1, max_k):
        for i in range(0,len(A[-1]['path'])):
            G0 = np.copy(G)
            edge_dev  = A[-1]['path'][i] # edge_dev[0] is the spur node
            path_root = A[-1]['path'][:i] # 0~i-1 until edge_dev[0]
            edges_removed = []
            for edge in path_root:
                G0[:,E[edge][0]] = Max
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i]:
                    G0[E[int(curr_path[i])][0]][E[int(curr_path[i])][1]] = Max
                    edges_removed.append(curr_path[i])
            
            D_1, p_1 = Shortest_Path(G0, EMap, E[int(edge_dev)][0], Max)
            
            if D_1[des] < Max:
                path_total = path_root + p_1[des]
                dis_total = len(path_root) + D_1[des]
                if dis_total != len(path_total):
                    print ([D_1[des], p_1[des], D[E[edge_dev][0]],path_root])
                potential_k = {'cost':dis_total,'path':path_total}
                if not (potential_k in B):
                    B.append(potential_k)
                 
        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break
    
    # ============== Check ====================
    # for B in A:
    #     path = B['path']
    #     SS = src
    #     for edge in path:
    #         edge_src = E[edge][0]
    #         if edge_src != SS:
    #             print('src error!')
    #         SS = E[edge][1]
    #     if SS!= des:
    #         print('des error!')
    return A
# def ksp_yen(graph, node_start, node_end, max_k=2):
#     distances, previous = dijkstra(graph, node_start)
    
#     A = [{'cost': distances[node_end], 
#           'path': path(previous, node_start, node_end)}]
#     B = []
    
#     if not A[0]['path']: return A
    
#     for k in range(1, max_k):
#         for i in range(0, len(A[-1]['path']) - 1):
#             node_spur = A[-1]['path'][i] #最短集合里最长的路径
#             path_root = A[-1]['path'][:i+1]
            
#             edges_removed = []
#             for path_k in A:
#                 curr_path = path_k['path']
#                 if len(curr_path) > i and path_root == curr_path[:i+1]:
#                     cost = graph.remove_edge(curr_path[i], curr_path[i+1])
#                     if cost == -1:
#                         continue
#                     edges_removed.append([curr_path[i], curr_path[i+1], cost])
            
#             path_spur = dijkstra(graph, node_spur, node_end)
            
#             if path_spur['path']:
#                 path_total = path_root[:-1] + path_spur['path']
#                 dist_total = distances[node_spur] + path_spur['cost']
#                 potential_k = {'cost': dist_total, 'path': path_total}
            
#                 if not (potential_k in B):
#                     B.append(potential_k)
            
#             for edge in edges_removed:
#                 graph.add_edge(edge[0], edge[1], edge[2])
        
#         if len(B):
#             B = sorted(B, key=itemgetter('cost'))
#             A.append(B[0])
#             B.pop(0)
#         else:
#             break
    
#     return A