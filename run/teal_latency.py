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
from tqdm import tqdm
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

        self.num_layer = 6

        self.edge_index = edge_index
        self.edge_index_values = edge_index_values
        self.num_path = 5
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
        self.num_path = 5
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


class ADMM():
    """Fine-tunes the allocations and mitigates constraint violations.
    F1_d = d - sum_{p in e} x_p - s1 for demand d;
    F3_e = c - sum_{pe in e} z_pe - s3 for edge e;
    F4_pe = x_p - z_pe for path p edge e;
    The augumented Lagranian for TE is
    L(x, z, s, lambda)
        = - sum_d sum_p x_p
        + lambda1 * F1 + lambda3 * F3 + lambda4 * F4
        + rho/2 * F1^2 + rho/2 * F3^2 + rho/2 * F4^2.
    """

    def __init__(
            self, p2e, num_path_node, num_edge_node, device
            ):
        """Initialize ADMM with the network topology.

        Args:
            dytop_env: dytop environment
            rho: hyperparameter for the augumented Lagranian
            device: device for new tensor to be allocated
        """

        self.rho = 1
        self.device = device

        self.p2e = p2e
        self.num_path = 5
        self.num_path_node = num_path_node
        self.num_edge_node = num_edge_node

        self.A_d_inv = self.init_A()
        self.b_extra = torch.zeros(self.num_edge_node).double().to(self.device)
        self.p2e_1_extra = torch.arange(self.num_edge_node).to(self.device)

    def init_A(self):
        """Return the inverse of A_d.
        A_d = diag(num_edges_for_paths) + 1 for demand d.
        """

        num_edges_for_path = torch_scatter.scatter(
            torch.ones(self.p2e.shape[1]).to(self.device),
            self.p2e[0])
        A_d_inv = torch.stack(
            [torch.linalg.inv(torch.diag(ele) + 1) for ele
                in num_edges_for_path.reshape(-1, self.num_path)])

        return A_d_inv

    def update_obs_action(self, obs, action):
        """Update demands, capacity, allocation."""

        # update demands and capacity
        self.d = obs[-self.num_path_node::self.num_path]
        self.c = obs[:-self.num_path_node]

        # update allocation in path-wise and edge-wise
        self.x = action
        self.z = self.x[self.p2e[0]]

        # init slack variables and lambda
        self.s1, self.s3 = 0, 0
        self.l1, self.l3, self.l4 = 0, 0, 0

    def update_admm(self):
        """Update ADMM for one round."""

        self.update_s()
        self.update_lambda()
        self.update_z()
        self.update_x()

    def update_s(self):
        """Update slack variables s = argmin_s L(x, z, s, lambda).
        s1 = - lambda1 / rho + (d - sum x_p)
        s3 = - lambda3 / rho + (c - sum z_pe)
        """

        self.s1 = self.l1 / self.rho \
            + (self.d - self.x.reshape(-1, self.num_path).sum(1))
        self.s3 = self.l3 / self.rho \
            + (self.c - torch_scatter.scatter(self.z, self.p2e[1], dim_size = self.num_edge_node))

        self.s1 = self.s1.relu()
        self.s3 = self.s3.relu()

    def update_x(self):
        """Update x = argmin_x L(x, z, s, lambda).
        x = - A_d_inv * b_d,
        where [b_d]_p = - 1 - lambda1_d + sum_{e in p} lambda4_pe
            + rho * (- d + s1_d) - rho * sum_{e in p} z_pe.
        """

        b = -1 - self.l1[:, None] \
            + self.rho*(-self.d + self.s1)[:, None]\
            + torch_scatter.scatter(
                self.l4-self.rho*self.z,
                self.p2e[0]).reshape(-1, self.num_path)

        self.x = -torch.einsum(
            "nab,nb->na",
            self.A_d_inv/self.rho,
            b).reshape(-1)

        # use x.relu() to approximate the non-negative solution
        self.x = self.x.relu()

    def update_z(self, num_approx=1):
        """Update z = argmin_z L(x, z, s, lambda).
        z = - A_e_inv * b_e,
        where [b_e]_p = - lambda3_e - lambda4_pe
            + rho * (- c_e + s_e - x_p).
        where A_e = I + 1.

        Args:
            num_approx: num of approx rounds for the non-negative solution
        """

        p2e_1 = self.p2e[1].clone()

        # 'double' precision is necessary:
        # torch_scatter is implemented via atomic operations on the GPU and is
        # therefore **non-deterministic** since the order of parallel
        # operations to the same value is undetermined.
        # For floating-point variables, this results in a source of variance in
        # the result.
        b = (
            self.rho*(
                -self.c[self.p2e[1]] + self.s3[self.p2e[1]]
                - self.x[self.p2e[0]])
            - self.l3[self.p2e[1]] - self.l4
        ).double()

        # z = - A_e_inv * b_e = sum b_e / (|b_e| + 1) - b
        # use b_extra and p2e_1_extra for |b_e| + 1
        b_mean = torch_scatter.scatter(
            torch.concat([b, self.b_extra]),
            torch.concat([p2e_1, self.p2e_1_extra]), reduce='mean')
        self.z = (b_mean[self.p2e[1]] - b)/self.rho

        # cannot use x.relu() to approximate the non-negative solution
        # iteratively decide which z is 0 and solve the rest of z
        for _ in range(num_approx):
            p2e_1[self.z < 0] = self.num_edge_node
            b_mean = torch_scatter.scatter(
                torch.concat([b, self.b_extra]),
                torch.concat([p2e_1, self.p2e_1_extra]), reduce='mean')
            self.z = (b_mean[self.p2e[1]] - b)/self.rho
        self.z = self.z.float().relu()

    def update_lambda(self):
        """Update lambda.
        lambda1 = lambda1 + rho * (d - sum_{p in e} x_p - s1);
        lambda3 = lambda3 + rho * (c - sum_{pe in e} z_pe - s3);
        lambda4 = lambda4 + rho * (x_p - z_pe).
        """

        self.l1 = self.l1 + self.rho * (
            self.d - self.x.reshape(-1, self.num_path).sum(1) - self.s1)
        self.l3 = self.l3 + self.rho * (
            self.c - torch_scatter.scatter(self.z, self.p2e[1], dim_size = self.num_edge_node) - self.s3)
        self.l4 = self.l4 + self.rho * (
            self.x[self.p2e[0]] - self.z)

    def tune_action(self, obs, action, num_admm_step):
        """Return fine-tuned allocations after ADMM.

        Args:
            obs: observation (capacity + traffic matrix)
            action: action to correct
            num_admm_step: number of admm steps
        """
        # init x, z, s, lambda
        self.update_obs_action(obs, action)

        # admm steps
        for _ in range(num_admm_step):
            self.update_admm()

        return self.x
    

def main(num_node, path_length):
    NUM_NODE = num_node
    PATH_LENGTH = path_length

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Create a list of available devices
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

    # If no GPUs are available, fall back to CPU
    if not devices:
        device = [torch.device('cpu')]

    print(devices)

    if NUM_NODE == 66:
        params = OrbitParams(
            GrdStationNum=0,
            Offset5=66,
            graph_node_num=132,
            isl_cap=200,
            uplink_cap=800,
            downlink_cap=800,
            ism=ISM.ISL,
        )
    elif NUM_NODE == 4000:
        params = OrbitParams(
            GrdStationNum=0,
            Offset5=4236,
            graph_node_num=8472,
            isl_cap=200,
            uplink_cap=800,
            downlink_cap=800,
            ism=ISM.ISL,
        )
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
    

    G = nx.DiGraph()
    G.add_nodes_from(range(params.graph_node_num))

    for i in range(params.graph_node_num):
        for j in range(params.graph_node_num):
            if i == j:
                continue
            G.add_edge(i, j, capacity=params.isl_cap)

    # edge nodes' degree, index lookup
    edge2idx_dict = {edge: idx for idx, edge in enumerate(G.edges)}
    node2degree_dict = {}
    edge_num = len(G.edges)
    num_path_node = (params.graph_node_num) * (params.graph_node_num - 1) * 5

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
    dst = [random.randint(0, edge_num-1) for _ in range(len(src))]

    edge_index_values = torch.tensor(
        [random.random()
            for u, v in zip(src+dst, dst+src)]).to(devices[0])
    edge_index = torch.tensor(
        [src+dst, dst+src], dtype=torch.long).to(devices[0])
    p2e = torch.tensor([src, dst], dtype=torch.long).to(devices[1])
    p2e[0] -= edge_num

    test_actor = TealActor(num_path_node, edge_index, edge_index_values, devices[0])
    admm = ADMM(p2e, num_path_node, edge_num, devices[1])

    tm = np.zeros((params.graph_node_num, params.graph_node_num))
    for i in range(params.graph_node_num):
        for j in range(params.graph_node_num):
            if i == j:
                continue
            tm[i][j] = random.randint(0, 100)

    tm = torch.FloatTensor(
            [[ele] * 5 for i, ele in enumerate(tm.flatten())
                if i % len(tm) != i//len(tm)]).flatten()
    capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in G.edges.data('capacity') if u != v])

    obs = torch.concat([capacity, tm]).to(devices[0])

    start = time.time()
    action = test_actor.act(obs)
    action = F.softmax(action, dim=-1).flatten() * tm.to(devices[0])
    model_time = time.time() - start
    action = action.to(devices[1])
    obs = obs.to(devices[1])
    admm.tune_action(obs, action, 5)
    runtime = time.time() - start
    admm_time = runtime - model_time

    # print(f'Total Latency of {NUM_NODE} Satellites: {runtime}')
    # print(f'Model Latency: {model_time}')
    # print(f'ADMM Latency: {admm_time}')

    return model_time, admm_time

if __name__ == "__main__":
    sizes = [66, 176, 500, 528, 1500, 4000][-2:]
    path_lens = [7, 9, 11, 12, 17, 25][-2:]
    for size, path_len in zip(sizes, path_lens):
        print(f'Running for {size} Satellites')
        main(size, path_len)
        total_model, total_admm = 0, 0
        for i in tqdm(range(2)):
            total_model += main(size, path_len)[0]
            total_admm += main(size, path_len)[1]

        with open('teal_latency.txt', 'a') as f:
            f.write(f'Latency for {size} Satellites with path length {path_len}:\n')
            f.write(f'Average Model Latency: {total_model/2}\n')
            f.write(f'Average ADMM Latency: {total_admm/2}\n')
            f.write(f'Average Total Latency: {(total_model + total_admm)/2}\n')