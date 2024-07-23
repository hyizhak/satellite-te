# from datasets import Dataset
import pickle
import sys
import numpy as np
import random
sys.path.append('../..')

from lib import AssetManager
from lib.data.starlink import StarlinkPathFormer, StarlinkAdapter, InterShellMode as ISM
from lib.data.starlink.orbit_params import OrbitParams
import networkx as nx
from lib.data.starlink.user_node import generate_sat2user


# for Ground Station
params_1 = lambda reduced: OrbitParams(
    GrdStationNum=222,
    Offset5=round(2 * 22 * 72 / reduced) if reduced != 0 else 4236,
    graph_node_num=round(2 * 22 * 72 / reduced) * 2 + 222 if reduced != 0 else 8694,
    isl_cap=200,
    uplink_cap=800,
    downlink_cap=800,
    ism=ISM.GRD_STATION,
)

# for ISL
params_2 = lambda reduced: OrbitParams(
    GrdStationNum=0,
    Offset5=round(2 * 22 * 72 / reduced) if reduced != 0 else 4236,
    graph_node_num=round(2 * 22 * 72 / reduced) * 2 if reduced != 0 else 8472,
    isl_cap=200,
    uplink_cap=800,
    downlink_cap=800,
    ism=ISM.ISL,
)

def construct_from_edge(edge_list, param, online=False, POP_ratio=1, size=None):

    if size is not None:
        match size:
            case 176:
                reduced = 18
            case 500:
                reduced = 8
            case 528:
                reduced = 6
            case 1500:
                reduced = 2
            case 5000:
                reduced = 0
    else:
        reduced = 0

    POP_ratio = float(POP_ratio)

    if param == 'ISL':
        params = params_2(reduced)
    else:
        params = params_1(reduced)

    """Construct a networkx graph from a list of edges."""
    sat2user = generate_sat2user(params.Offset5, params.GrdStationNum, params.ism)
    G = nx.DiGraph()
    G.add_nodes_from(range(params.graph_node_num))

    ## 1. Inter-satellite links
    if params.ism == ISM.ISL:
        for i, e in enumerate(edge_list):
            if i < 16943:
                G.add_edge(e[0], e[1], capacity=params.isl_cap/POP_ratio)
            else :
                if random.random() < 0.325 and online:
                    G.add_edge(e[0], e[1], capacity=0)
                else:
                    G.add_edge(e[0], e[1], capacity=params.isl_cap/POP_ratio)
    else:
        for i, e in enumerate(edge_list):
            if i < 16943:
                G.add_edge(e[0], e[1], capacity=params.isl_cap/POP_ratio)
            else :
                if random.random() < 0.675 and online:
                    G.add_edge(e[0], e[1], capacity=0)
                else:
                    G.add_edge(e[0], e[1], capacity=params.isl_cap/POP_ratio)

    ## 2. User-satellite links
    for i in range(params.Offset5):
        # Uplink
        G.add_edge(sat2user(i), i, capacity=params.uplink_cap/POP_ratio)
        # Downlink
        G.add_edge(i, sat2user(i), capacity=params.downlink_cap/POP_ratio)

    ## 3. Inter ground station links
    for i in range(params.GrdStationNum):
        for j in range(params.GrdStationNum):
            if i == j:
                continue
            G.add_edge(i + params.Offset5, j + params.Offset5, capacity=0)
            G.add_edge(j + params.Offset5, i + params.Offset5, capacity=0)

    ## 4. User-ground station links
    for i in range(params.Offset5):
        for j in range(params.GrdStationNum):
            if not G.has_edge(sat2user(i), j + params.Offset5):
                G.add_edge(sat2user(i), j + params.Offset5, capacity=0)
            if not G.has_edge(j + params.Offset5, sat2user(i)):
                G.add_edge(j + params.Offset5, sat2user(i), capacity=0)

    ## 5. Satellite-ground station links
    for i in range(params.Offset5):
        for j in range(params.GrdStationNum):
            if not G.has_edge(i, j + params.Offset5):
                G.add_edge(i, j + params.Offset5, capacity=0)
            if not G.has_edge(j + params.Offset5, i):
                G.add_edge(j + params.Offset5, i, capacity=0)

    ## 6. Inter-user links
    for i in range(params.Offset5):
        for j in range(params.Offset5):
            if i == j:
                continue
            if not G.has_edge(sat2user(i), sat2user(j)):
                G.add_edge(sat2user(i), sat2user(j), capacity=0)
            if not G.has_edge(sat2user(j), sat2user(i)):
                G.add_edge(sat2user(j), sat2user(i), capacity=0)


    return G

def dict_to_numpy(dictionary, param, size=None):

    if size is not None:
        match size:
            case 176:
                reduced = 18
            case 500:
                reduced = 8
            case 528:
                reduced = 6
            case 1500:
                reduced = 2
            case 5000:
                reduced = 0
    else:
        reduced = 0

    if param == 'ISL':
        params = params_2(reduced)
    else:
        params = params_1(reduced)

    # init a 2d array of 0s
    array = np.zeros((params.graph_node_num, params.graph_node_num))
    # iterate over the dictionary
    for key, value in dictionary.items():
        src, dst = key.split(', ')
        # fill the array with flows
        array[int(src), int(dst)] = value
    return array

def generate_random_values(k, total_sum):
    # Generate k random values from the Dirichlet distribution
    random_values = np.random.dirichlet(np.ones(k))
    # Scale the values so that their summation equals the total_sum
    scaled_values = random_values * total_sum
    return scaled_values

def split_tm(tm, k):
    tm = tm.astype(float)
    allocated_arrays = [np.zeros_like(tm) for _ in range(k)]

    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            allocated_values = generate_random_values(k, tm[i, j])
            for idx, value in enumerate(allocated_values):
                allocated_arrays[idx][i, j] = value

    return allocated_arrays
