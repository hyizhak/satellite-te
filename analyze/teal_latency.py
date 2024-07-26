import math
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
from contextlib import contextmanager

import torch_scatter
import torch_sparse
import networkx as nx

from lib.data.starlink.orbit_params import OrbitParams
from lib.data.starlink.ism import InterShellMode as ISM
from lib.data.starlink.user_node import generate_sat2user


def weight_initialization(module):
    """Initialize weights in nn module"""

    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


def uni_rand(low=-1, high=1):
    """Uniform random variable [low, high)"""
    return (high - low) * np.random.rand() + low


def print_(*args, file=None):
    """print out *args to file"""
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()


class FlowGNN(nn.Module):
    """Transform the demands into compact feature vectors known as embeddings.

    FlowGNN alternates between
    - GNN layers aimed at capturing capacity constraints;
    - DNN layers aimed at capturing demand constraints.

    Replace torch_sparse package with torch_geometric pakage is possible
    but require larger memory space.
    """

    def __init__(self, edge_index, edge_index_values, num_path_node):
        """Initialize flowGNN with the network topology.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
        """

        super(FlowGNN, self).__init__()

        self.num_layer = 1

        self.edge_index = edge_index
        self.edge_index_values = edge_index_values
        self.num_path = 4
        self.num_path_node = num_path_node
        # self.adj_adj = torch.sparse_coo_tensor(self.edge_index,
        #    self.edge_index_values,
        #    [self.num_path_node + self.num_edge_node,
        #    self.num_path_node + self.num_edge_node])

        self.gnn_list = []
        self.dnn_list = []
        for i in range(self.num_layer):
            # to replace with GCNConv package:
            # self.gnn_list.append(GCNConv(i+1, i+1))
            self.gnn_list.append(nn.Linear(i+1, i+1))
            self.dnn_list.append(
                nn.Linear(self.num_path*(i+1), self.num_path*(i+1)))
        self.gnn_list = nn.ModuleList(self.gnn_list)
        self.dnn_list = nn.ModuleList(self.dnn_list)

        # weight initialization for dnn and gnn
        self.apply(weight_initialization)

    def forward(self, h_0):
        """Return embeddings after forward propagation

        Args:
            h_0: inital embeddings
        """

        h_i = h_0
        for i in range(self.num_layer):

            # gnn
            # to replace with GCNConv package:
            # h_i = self.gnn_list[i](h_i, self.edge_index)
            h_i = self.gnn_list[i](h_i)
            # h_i = torch.sparse.mm(self.adj_adj, h_i)
            h_i = torch_sparse.spmm(
                self.edge_index, self.edge_index_values,
                h_0.shape[0], h_0.shape[0], h_i)

            # dnn
            h_i_path_node = self.dnn_list[i](
                h_i[-self.num_path_node:, :].reshape(
                    self.num_path_node//self.num_path,
                    self.num_path*(i+1)))\
                .reshape(self.num_path_node, i+1)
            h_i = torch.concat(
                [h_i[:-self.num_path_node, :], h_i_path_node], axis=0)

            # skip connection
            h_i = torch.cat([h_i, h_0], axis=-1)

        # return path-node embeddings
        return h_i[-self.num_path_node:, :]


class TealActor(nn.Module):

    def __init__(
            self, num_path_node, edge_index, edge_index_values, device,
            std=1, log_std_min=-10.0, log_std_max=10.0):
        """Initialize teal actor.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
            model_dir: model save directory
            model_save: whether to save the model
            device: device id
            std: std value, -1 if apply neuro networks for std
            log_std_min: lower bound for log std
            log_std_max: upper bound for log std
        """

        super(TealActor, self).__init__()

        # teal environment
        self.num_path = 4
        self.num_path_node = num_path_node

        # init FlowGNN
        self.device = device
        self.FlowGNN = FlowGNN(edge_index, edge_index_values, num_path_node).to(self.device)

        # init COMA policy
        self.std = std
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.mean_linear = nn.Linear(
            self.num_path*(self.FlowGNN.num_layer+1),
            self.num_path).to(self.device)
        # apply neuro networks for std
        if std < 0:
            self.log_std_linear = nn.Linear(
                self.num_path*(self.FlowGNN.num_layer+1),
                self.num_path).to(self.device)

        # init model
        self.apply(weight_initialization)

    def forward(self, feature):
        """Return mean of normal distribution after forward propagation

        Args:
            features: input features including capacity and demands
        """
        x = self.FlowGNN(feature)
        x = x.reshape(
            self.num_path_node//self.num_path,
            self.num_path*(self.FlowGNN.num_layer+1))
        mean = self.mean_linear(x)

        # to apply neuro networks for std
        if self.std < 0:
            log_std = self.log_std_linear(x)
            log_std_clamped = torch.clamp(
                log_std,
                min=self.log_std_min,
                max=self.log_std_max)
            nn_std = torch.exp(log_std_clamped)
            return mean, nn_std
        # deterministic std
        else:
            return mean, self.std

    def evaluate(self, obs, deterministic=False):
        """Return raw action before softmax split ratio.

        Args:
            obs: input features including capacity and demands
            deterministic: whether to have deterministic action
        """

        feature = obs.reshape(-1, 1)
        mean, std = self.forward(feature)
        mean = mean

        # test mode
        if deterministic:
            distribution = None
            raw_action = mean.detach()
            log_probability = None
        # train mode
        else:
            # use normal distribution for action
            distribution = Normal(mean, std)
            sample = distribution.rsample()
            raw_action = sample.detach()
            log_probability = distribution.log_prob(raw_action).sum(axis=-1)

        return raw_action, log_probability

    def act(self, obs):
        """Return split ratio as action with disabled gradient calculation.

        Args:
            obs: input observation including capacity and demands
        """

        with torch.no_grad():
            # deterministic action
            raw_action, _ = self.evaluate(obs, deterministic=True)
            return raw_action

def main(num_node, path_length):
    NUM_NODE = num_node
    PATH_LENGTH = path_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if NUM_NODE == 4000:
        params = OrbitParams(
            GrdStationNum=0,
            Offset5=4236,
            graph_node_num=8472,
            isl_cap=200,
            uplink_cap=800,
            downlink_cap=800,
            ism=ISM.ISL,
        )
        graph_path = '/data/projects/11003765/sate/input/starlink/DataSetForSaTE100/ISL_teal/topo_0/graph_edge.pickle'
        tm_path = '/data/projects/11003765/sate/input/starlink/DataSetForSaTE100/ISL_teal/topo_0/tm_train/0.pkl'
    else:
        match NUM_NODE:
            case 176:
                reduced = 18
            case 500:
                reduced = 8
            case 528:
                reduced = 6
            case 1500:
                reduced = 2
        params = OrbitParams(
            GrdStationNum=0,
            Offset5=round(2 * 22 * 72 / reduced),
            graph_node_num=round(2 * 22 * 72 / reduced) * 2,
            isl_cap=200,
            uplink_cap=800,
            downlink_cap=800,
            ism=ISM.ISL,
        )
        graph_path = f'/data/projects/11003765/sate/input/starlink/starlink_{NUM_NODE}/ISL_teal/topo_0/graph_edge.pickle'
        tm_path = f'/data/projects/11003765/sate/input/starlink/starlink_{NUM_NODE}/ISL_teal/topo_0/tm_train/0.pkl'
        

    # load graph edges
    with open(graph_path, 'rb') as f:
        E = pickle.load(f)

    sat2user = generate_sat2user(params.Offset5, params.GrdStationNum, params.ism)
    G = nx.DiGraph()
    G.add_nodes_from(range(params.graph_node_num))

    ## 1. Inter-satellite links
    for e in E:
        G.add_edge(e[0], e[1], capacity=params.isl_cap)

    ## 2. User-satellite links
    for i in range(params.Offset5):
        # Uplink
        G.add_edge(sat2user(i), i, capacity=params.uplink_cap)
        # Downlink
        G.add_edge(i, sat2user(i), capacity=params.downlink_cap)

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
            G.add_edge(sat2user(i), j + params.Offset5, capacity=0)
            G.add_edge(j + params.Offset5, sat2user(i), capacity=0)

    ## 5. Satellite-ground station links
    for i in range(params.Offset5):
        for j in range(params.GrdStationNum):
            # if G.has_edge(i, j + params.Offset5):
            #     continue
            G.add_edge(i, j + params.Offset5, capacity=0)
            G.add_edge(j + params.Offset5, i, capacity=0)

    ## 6. Inter-user links
    for i in range(params.Offset5):
        for j in range(params.Offset5):
            if i == j:
                continue
            G.add_edge(sat2user(i), sat2user(j), capacity=0)
            G.add_edge(sat2user(j), sat2user(i), capacity=0)

    # edge nodes' degree, index lookup
    edge2idx_dict = {edge: idx for idx, edge in enumerate(G.edges)}
    node2degree_dict = {}
    edge_num = len(G.edges)
    num_path_node = NUM_NODE * (NUM_NODE - 1)

    # build edge_index
    src, dst, path_i = [], [], 0
    # for s in tqdm(range(len(G))):
    #     for t in range(len(G)):
    #         if s == t:
    #             continue
    #         for path in path_dict.get((s, t), [[s, t] for _ in range(num_path)]):
    #             for (u, v) in zip(path[:-1], path[1:]):
    #                 src.append(edge_num+path_i)
    #                 dst.append(edge2idx_dict[(u, v)])

    #                 if src[-1] not in node2degree_dict:
    #                     node2degree_dict[src[-1]] = 0
    #                 node2degree_dict[src[-1]] += 1
    #                 if dst[-1] not in node2degree_dict:
    #                     node2degree_dict[dst[-1]] = 0
    #                 node2degree_dict[dst[-1]] += 1
    #             path_i += 1

    src = [i for i in range(edge_num, edge_num + num_path_node) for _ in range(PATH_LENGTH)]
    dst = [random.randint(0, edge_num) for _ in range(len(src))]

    edge_index_values = torch.tensor(
        [random.random()
            for u, v in zip(src+dst, dst+src)]).to(device)
    edge_index = torch.tensor(
        [src+dst, dst+src], dtype=torch.long).to(device)

    test_actor = TealActor(num_path_node, edge_index, edge_index_values, device)

    # load traffic matrix
    with open(tm_path, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct the traffic matrix
    size = data['size']
    edge_list = data['edge_list']
    weight_list = data['weight_list']
    
    tm = np.zeros((size, size))
    for edge, weight in zip(edge_list, weight_list):
        tm[edge[0], edge[1]] = weight

    tm = torch.FloatTensor(
            [[ele] * 4 for i, ele in enumerate(tm.flatten())
                if i % len(tm) != i//len(tm)]).flatten()
    capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in G.edges.data('capacity') if u != v])

    obs = torch.concat([capacity, tm]).to(device)

    start = time.time()
    test_actor.act(obs)
    runtime = time.time() - start

    print(f'Latency of {NUM_NODE} Satellites: {runtime}')

    return runtime

if __name__ == "__main__":
    main(1500, 4)
    total = 0
    for i in range(10):
        total += main(1500, 4)

    print(f'Average Latency: {total/10}')