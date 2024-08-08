import pickle
import json
import os
import math
import time
import random
import numpy as np
import networkx as nx
from itertools import product

from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split
from pathlib import Path
import dgl

import torch
import torch_scatter
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
from lib.data.starlink.user_node import generate_sat2user
from lib.data.starlink.orbit_params import OrbitParams

from .. import AssetManager
from .ADMM import ADMM
from .path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles


class SaTEEnv(object):

    def __init__(
            self, obj, problem_path,
            num_path, edge_disjoint, dist_metric, rho,
            num_failure, device,
            work_dir, dataset, supervised,
            orbit_params=None,
            raw_action_min=-10.0, raw_action_max=10.0):
        """Initialize SaTE environment.

        Args:
            obj: objective
            topo: topology name
            problems: problem list
            num_path: number of paths per demand
            edge_disjoint: whether edge-disjoint paths
            dist_metric: distance metric for shortest paths
            rho: hyperparameter for the augumented Lagranian
            train size: train start index, stop index
            val size: val start index, stop index
            test size: test start index, stop index
            device: device id
            raw_action_min: min value when clamp raw action
            raw_action_max: max value when clamp raw action
        """

        self.obj = obj
        self.orbit_params = orbit_params
        self.problem_path = problem_path
        self.num_path = num_path
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric
        
        self.work_dir = work_dir

        self.num_failure = num_failure
        self.device = device

        self.supervised = supervised

        # # init matrices related to topology
        # self.G = self._read_graph_json(topo)
        # self.capacity = torch.FloatTensor(
        #     [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        # self.num_edge_node = len(self.G.edges)
        # self.num_path_node = self.num_path * self.G.number_of_nodes()\
        #     * (self.G.number_of_nodes()-1)
        # self.edge_index, self.edge_index_values, self.p2e = \
        #     self.get_topo_matrix(topo, num_path, edge_disjoint, dist_metric)

        # # init ADMM
        # self.ADMM = ADMM(
        #     self.p2e, self.num_path, self.num_path_node,
        #     self.num_edge_node, rho, self.device)

        self.rho = rho

        # min/max value when clamp raw action
        self.raw_action_min = raw_action_min
        self.raw_action_max = raw_action_max
        
        # prepare the dataset
        if self.supervised:
            self.train_dataset, self.test_dataset, self.train_label, self.test_label = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)
        else:
            self.train_dataset, self.test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

        self.mode = None
        self.reset('train')

    def reset(self, mode='test'):
        """Reset the initial conditions in the beginning."""
        
        self.mode = mode

        if (mode == 'train' or mode == 'validate'):
            self.dataset = self.train_dataset
        elif mode == 'test':
            self.dataset = self.test_dataset

        self.idx_stop = len(self.dataset)

        self.idx = 0
        self.obs = self._read_obs()

    def get_obs(self):
        """Return observation (capacity + traffic matrix)."""

        return self.obs

    def _read_obs(self):
        """Return observation (capacity + traffic matrix) from files."""

        # {'graph': E, 'tm': tm_dict, 'path': path_dict, 'data_idx': data_idx}
        data = self.dataset[self.idx]

        # init matrices related to topology
        self.G = self.construct_from_edge(data['graph'])
        # self.capacity = torch.FloatTensor(
        #     [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)
        
        # src, dst, capacity = [], [], []
        # for u, v, data in self.G.edges(data=True):
        #     src.append(u)
        #     dst.append(v)
        #     capacity.append(data['capacity'])

        # capacity = torch.FloatTensor(capacity).to(self.device)

        # # Create a DGL graph from the node tensor
        # self.G_dgl = dgl.graph((torch.tensor(src), torch.tensor(dst))).to(self.device)

        # # Add edge data (capacity) to the DGL graph
        # self.G_dgl.edata['capacity'] = capacity

        # Extract edges and capacities
        src, dst, capacities = zip(*[(u, v, edata['capacity']) 
                                    for u, v, edata in self.G.edges(data=True)])

        # Convert to PyTorch tensors
        src_tensor = torch.tensor(src, dtype=torch.int64)
        dst_tensor = torch.tensor(dst, dtype=torch.int64)
        self.capacities_tensor = torch.tensor(capacities, dtype=torch.float32).to(self.device)
    
        # Create a DGLGraph
        self.G_dgl = dgl.graph((src_tensor, dst_tensor)).to(self.device)
        # Add edge data (capacity)
        self.G_dgl.edata['capacity'] = self.capacities_tensor

        problem_G = self.create_heterograph(data)

        # Add node data
        self.G_dgl.ndata['in_traffic'] = self.in_traffic
        self.G_dgl.ndata['out_traffic'] = self.out_traffic    
        
        # remove demands within nodes and 0 demands
        tm = torch.FloatTensor(self.flow_values).flatten().to(self.device)
        tm = torch.repeat_interleave(tm, self.num_path)
        # obs = torch.concat([self.capacity, tm]).to(self.device)
        obs = {"topo": self.G_dgl,
               "capacity": self.capacities_tensor,
               "traffic": tm,
               "problem": problem_G}
        return obs

    def _next_obs(self):
        """Return next observation (capacity + traffic matrix)."""

        self.idx += 1
        if self.idx == self.idx_stop:
            self.idx = 0
        self.obs = self._read_obs()
        return self.obs

    def render(self):
        """Return a dictionary for the details of the current problem"""

        problem_dict = {
            'problem_path': self.problem_path,
            'obj': self.obj,
            'topo_idx': self.idx,
            'tm_idx': self.idx,
            'num_node': self.G.number_of_nodes(),
            'num_edge': self.G.number_of_edges(),
            'num_path': self.num_path,
            'edge_disjoint': self.edge_disjoint,
            'dist_metric': self.dist_metric,
            'total_demand': self.obs['traffic'][::self.num_path].sum().item(),
        }
        return problem_dict

    def step(self, raw_action, num_sample=0, num_admm_step=0):
        """Return the reward of current action.

        Args:
            raw_action: raw action from actor
            num_sample: number of samples for reward during training
            num_admm_step: number of ADMM steps during testing
        """

        info = {}
        if self.mode == 'train':
            reward = self.take_action(raw_action, num_sample)
        else:
            start_time = time.time()
            action = self.transform_raw_action(raw_action)
            if self.obj.endswith('total_flow'):
                # total flow require no constraint violation
                obs = torch.concat([self.obs['capacity'], self.obs['traffic']]).to(self.device)
                action = self.ADMM.tune_action(obs, action, num_admm_step)
                action = self.round_action(action)
            info['runtime'] = time.time() - start_time
            info['sol_mat'] = self.extract_sol_mat(action)
            reward = self.get_obj(action)

        # next observation
        self._next_obs()
        return reward, info
    
    def compute_loss(self, raw_action):
        
        assert self.supervised and self.mode == 'train'

        action = self.transform_raw_action(raw_action)

        labels = self.train_label[self.idx]

        # Adding a small value epsilon to avoid zeros in the labels
        epsilon = 1e-10
        labels = labels + epsilon

        # Renormalize to ensure the distributions still sum to 1
        labels = labels / labels.sum(dim=1, keepdim=True)

        # Compute the log of the model's output probabilities
        log_output = action.log()

        # Compute KL divergence
        loss = F.kl_div(log_output, labels, reduction='batchmean')

        return loss


    def get_obj(self, action):
        """Return objective."""

        if self.obj.endswith('total_flow'):
            return action.sum(axis=-1)
        elif self.obj == 'min_max_link_util':
            return (torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1]
                )/self.obs['capacity']).max()

    def transform_raw_action(self, raw_action):
        """Return network flow allocation as action.

        Args:
            raw_action: raw action directly from ML output
        """
        # clamp raw action between raw_action_min and raw_action_max
        raw_action = torch.clamp(
            raw_action, min=self.raw_action_min, max=self.raw_action_max)

        # translate ML output to split ratio through softmax
        # 1 in softmax represent unallocated traffic
        raw_action = raw_action.exp()
        raw_action = raw_action/(1+raw_action.sum(axis=-1)[:, None])

        if self.supervised:
            return raw_action

        # translate split ratio to flow
        raw_action = raw_action.flatten() * self.obs['traffic']

        return raw_action

    def round_action(
            self, action, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """Return rounded action.
        Action can still violate constraints even after ADMM fine-tuning.
        This function rounds the action through cutting flow.

        Args:
            action: input action
            round_demand: whether to round action for demand constraints
            round_capacity: whether to round action for capacity constraints
            num_round_iter: number of rounds when iteratively cutting flow
        """

        demand = self.obs['traffic'][::self.num_path]
        capacity = self.obs['capacity']

        # reduce action proportionally if action exceed demand
        if round_demand:
            action = action.reshape(-1, self.num_path)
            ratio = action.sum(-1) / demand
            action[ratio > 1, :] /= ratio[ratio > 1, None]
            action = action.flatten()

        # iteratively reduce action proportionally if action exceed capacity
        if round_capacity:
            path_flow = action
            path_flow_allocated_total = torch.zeros(path_flow.shape)\
                .to(self.device)
            for round_iter in range(num_round_iter):
                # flow on each edge
                edge_flow = torch_scatter.scatter(
                    path_flow[self.p2e[0]], self.p2e[1], dim_size = self.num_edge_node)
                # util of each edge
                util = 1 + (edge_flow/capacity-1).relu()
                # propotionally cut path flow by max util
                util = torch_scatter.scatter(
                    util[self.p2e[1]], self.p2e[0], reduce="max")
                path_flow_allocated = path_flow/util
                # update total allocation, residual capacity, residual flow
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[self.p2e[0]], self.p2e[1], dim_size = self.num_edge_node)).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total

        return action

    def take_action(self, raw_action, num_sample):
        '''Return an approximate reward for action for each node pair.
        To make function fast and scalable on GPU, we only calculate delta.
        We assume when changing action in one node pair:
        (1) The change in edge utilization is very small;
        (2) The bottleneck edge in a path does not change due to (1).
        For evary path after change:
            path_flow/max(util, 1) =>
            (path_flow+delta_path_flow)/max(util+delta_util, 1)
            if util < 1:
                reward = - delta_path_flow
            if util > 1:
                reward = - delta_path_flow/(util+delta_util)
                    + path_flow*delta_util/(util+delta_util)/util
                    approx delta_path_flow/util - path_flow/util^2*delta_util

        Args:
            raw_action: raw action from policy network
            num_sample: number of samples in estimating reward
        '''

        path_flow = self.transform_raw_action(raw_action)

        if self.obj == 'rounded_total_flow':
            path_flow = self.round_action(path_flow)

        edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1], dim_size = self.num_edge_node)
        util = edge_flow/self.obs['capacity']

        # sample from uniform distribution [mean_min, min_max]
        distribution = Uniform(
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_min,
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_max)
        reward = torch.zeros(self.num_path_node//self.num_path).to(self.device)

        if self.obj == 'teal_total_flow' or self.obj == 'rounded_total_flow':

            # find bottlenack edge for each path
            util, path_bottleneck = torch_scatter.scatter_max(
                util[self.p2e[1]], self.p2e[0])
            path_bottleneck = self.p2e[1][path_bottleneck]

            # prepare -path_flow/util^2 for reward
            coef = path_flow/util**2
            coef[util < 1] = 0
            coef = torch_scatter.scatter(
                coef, path_bottleneck, dim_size=self.num_edge_node).reshape(-1, 1)

            # prepare path_util to bottleneck edge_util
            bottleneck_p2e = torch.sparse_coo_tensor(
                self.p2e, (1/self.obs['capacity'])[self.p2e[1]],
                [self.num_path_node, self.num_edge_node])

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                # add -delta_path_flow if util < 1 else -delta_path_flow/util
                delta_path_flow = self.transform_raw_action(sample) - path_flow
                reward += -(delta_path_flow/(1+(util-1).relu()))\
                    .reshape(-1, self.num_path).sum(-1)

                # add path_flow/util^2*delta_util for each path
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                # get utilization changes on edge
                # do not use torch_sparse.spspmm()
                # "an illegal memory access was encountered" in large topology
                delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
                reward += torch.sparse.mm(delta_util, coef).flatten()

        elif self.obj == 'teal_min_max_link_util':

            # find link with max utilization
            max_util_edge = util.argmax()

            # prepare paths related to max_util_edge
            max_util_paths = torch.zeros(self.num_path_node).to(self.device)
            max_util_paths[self.p2e[0, self.p2e[1] == max_util_edge]] =\
                1/self.obs[max_util_edge]

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                delta_path_flow = self.transform_raw_action(sample) - path_flow
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                reward += torch.sparse.mm(
                    delta_path_flow, max_util_paths.reshape(-1, 1)).flatten()
        
        elif self.obj == 'total_flow':
            reward = path_flow.sum(axis=-1)
            penalty = edge_flow - self.obs['capacity']
            reward -= penalty.relu().sum()

        return reward if self.obj == 'total_flow' else reward/num_sample 

    def read_graph(self, topo):
        """Return network topo from json file."""

        return nx.read_gpickle(topo)

    # def path_full_fname(self, topo, num_path, edge_disjoint, dist_metric):
    #     """Return full name of the topology path."""

    #     return os.path.join(
    #         TOPOLOGIES_DIR, "paths", "path-form",
    #         "{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
    #             topo, num_path, edge_disjoint, dist_metric))

    def get_path(self, num_path, edge_disjoint, dist_metric):
        """Return path dictionary."""

        graph_path = self.dataset[self.idx // self.num_tm][0]
        
        self.path_fname = path_fname = AssetManager.pathform_path(graph_path, num_path, edge_disjoint, dist_metric)
        
        # print("Loading paths from pickle file", path_fname)
        try:
            with open(path_fname, 'rb') as f:
                path_dict = pickle.load(f)
                # print("path_dict size:", len(path_dict))
                return path_dict
        except FileNotFoundError:
            # print("Creating paths {}".format(path_fname))
            path_dict = self.compute_path(num_path, edge_disjoint, dist_metric)
            # print("Saving paths to pickle file")
            with open(path_fname, "wb") as w:
                pickle.dump(path_dict, w)
        return path_dict

    def compute_path(self, num_path, edge_disjoint, dist_metric):
        """Return path dictionary through computation."""

        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_path, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def get_regular_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary with the same number of paths per demand.
        Fill with the first path when number of paths is not enough.
        """

        path_dict = self.get_path(num_path, edge_disjoint, dist_metric)
        for (s_k, t_k) in path_dict:
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [
                    path_dict[(s_k, t_k)][0] for _
                    in range(self.num_path - len(path_dict[(s_k, t_k)]))]\
                    + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    # def get_topo_matrix(self, topo, num_path, edge_disjoint, dist_metric):
    #     """
    #     Return matrices related to topology.
    #     edge_index, edge_index_values: index and value for matrix
    #     D^(-0.5)*(adjacent)*D^(-0.5) without self-loop
    #     p2e: [path_node_idx, edge_nodes_inx]
    #     """

    #     # get regular path dict
    #     path_dict = self.get_regular_path(
    #         topo, num_path, edge_disjoint, dist_metric)
        
    #     self.paths = path_dict

    #     # edge nodes' degree, index lookup
    #     edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
    #     node2degree_dict = {}
    #     edge_num = len(self.G.edges)

    #     # build edge_index
    #     src, dst, path_i = [], [], 0
    #     for s in range(len(self.G)):
    #         for t in range(len(self.G)):
    #             if s == t:
    #                 continue
    #             for path in path_dict[(s, t)]:
    #                 for (u, v) in zip(path[:-1], path[1:]):
    #                     src.append(edge_num+path_i)
    #                     dst.append(edge2idx_dict[(u, v)])

    #                     if src[-1] not in node2degree_dict:
    #                         node2degree_dict[src[-1]] = 0
    #                     node2degree_dict[src[-1]] += 1
    #                     if dst[-1] not in node2degree_dict:
    #                         node2degree_dict[dst[-1]] = 0
    #                     node2degree_dict[dst[-1]] += 1
    #                 path_i += 1

    #     # edge_index is D^(-0.5)*(adj)*D^(-0.5) without self-loop
    #     edge_index_values = torch.tensor(
    #         [1/math.sqrt(node2degree_dict[u]*node2degree_dict[v])
    #             for u, v in zip(src+dst, dst+src)]).to(self.device)
    #     edge_index = torch.tensor(
    #         [src+dst, dst+src], dtype=torch.long).to(self.device)
    #     p2e = torch.tensor([src, dst], dtype=torch.long).to(self.device)
    #     p2e[0] -= len(self.G.edges)

    #     return edge_index, edge_index_values, p2e
    
    def create_heterograph(self, data):

        # # Initialize lists to store source, destination, and flow values
        # src, dst, flow_values = [], [], []

        # # Iterate through the flow array to extract non-zero flows
        # for i in range(tm.shape[0]):
        #     for j in range(tm.shape[1]):
        #         if tm[i][j] != 0 and i != j:
        #             src.append(i)
        #             dst.append(j)
        #             flow_values.append(tm[i][j])

        # sort the traffic matrix by src, dst
        sorted_tm = sorted(
            data['tm'].items(),
            key=lambda item: (int(item[0].split(', ')[0]), int(item[0].split(', ')[1]))
        )

        # Extract src, dst, and flow_values
        src = [int(key.split(', ')[0]) for key, _ in sorted_tm]
        dst = [int(key.split(', ')[1]) for key, _ in sorted_tm]
        flow_values = [value for _, value in sorted_tm]
        self.flow_values = flow_values

        num_nodes = self.G_dgl.num_nodes()
        self.in_traffic = torch.zeros(num_nodes).to(self.device)
        self.out_traffic = torch.zeros(num_nodes).to(self.device)

        # Accumulate traffic
        for s, d, flow in zip(src, dst, flow_values):
            self.out_traffic[s] += flow
            self.in_traffic[d] += flow

        # src, dst = np.nonzero(np.where(np.eye(tm.shape[0]) == 1, 0, tm))
        # flow_values = tm[src, dst]
        # self.flow_values = flow_values

        self.num_path_node = self.num_path * len(flow_values)
        # self.edge_index, self.edge_index_values, self.p2e = \
        #         self.get_topo_matrix(topo, self.num_path, self.edge_disjoint, self.dist_metric)
        
        paths = data['path']
        self.paths = paths

        # edge nodes' degree, index lookup
        edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
        node2degree_dict = {}
        edge_num = len(self.G.edges)

        flow_use_path = [[], []]
        flow_count = 0
        src_list, dst_list = [], []

        path_values = [0] * self.num_path_node
        for (src, dst) in zip(src, dst):
            configured_paths = paths.get(f'{src}, {dst}', [])
            flow_use_path[0] += [flow_count] * len(configured_paths)
            # index = self.num_path * ((self.G.number_of_nodes() - 1) * src + dst) if src > dst \
            #     else self.num_path * ((self.G.number_of_nodes() - 1) * src + dst - 1)
            index = flow_count * self.num_path
            flow_use_path[1] += list(range(index, index + self.num_path))
            flow_count += 1

            for i, path in enumerate(configured_paths):
                path_values[index+i] = len(path)
                path_i = index + i

                for (u, v) in zip(path[:-1], path[1:]):
                    src_list.append(edge_num+path_i)
                    dst_list.append(edge2idx_dict[(u, v)])

                    if src_list[-1] not in node2degree_dict:
                        node2degree_dict[src_list[-1]] = 0
                    node2degree_dict[src_list[-1]] += 1
                    if dst_list[-1] not in node2degree_dict:
                        node2degree_dict[dst_list[-1]] = 0
                    node2degree_dict[dst_list[-1]] += 1

        # edge_index is D^(-0.5)*(adj)*D^(-0.5) without self-loop
        self.edge_index_values = torch.tensor(
            [1/math.sqrt(node2degree_dict[u]*node2degree_dict[v])
                for u, v in zip(src_list+dst_list, dst_list+src_list)]).to(self.device)
        self.edge_index = torch.tensor(
            [src_list+dst_list, dst_list+src_list], dtype=torch.long).to(self.device)
        p2e = torch.tensor([src_list, dst_list], dtype=torch.long).to(self.device)
        p2e[0] -= edge_num
        self.p2e = p2e
        e2p = torch.tensor([dst_list, src_list], dtype=torch.long).to(self.device)
        e2p[1] -= edge_num

        self.ADMM = ADMM(
                self.p2e, self.num_path, self.num_path_node,
                self.num_edge_node, self.rho, self.device)

        flow_use_path = tuple(torch.tensor(sublist).to(self.device) for sublist in flow_use_path)

        link_constitute_path = tuple(e2p)

        graph_data = {
            ('flow', 'uses', 'path'): flow_use_path,
            ('link', 'constitutes', 'path'): link_constitute_path,
            }
        
        num_nodes_dict = {'flow': len(flow_values), 
                      'path': self.num_path_node,
                      'link': self.num_edge_node}
        
        # print(num_nodes_dict)

        G = dgl.heterograph(data_dict=graph_data, num_nodes_dict=num_nodes_dict)

        G.nodes['flow'].data['x'] = torch.Tensor(flow_values).to(self.device)
        G.nodes['path'].data['x'] = torch.Tensor(path_values).to(self.device)

        return G
    

    def construct_from_edge(self, edge_list):
        params = self.orbit_params

        """Construct a networkx graph from a list of edges."""

        path = Path(self.problem_path)

        if len(path.parts) > 1 and path.parts[-3] == 'starlink':
            sat2user = generate_sat2user(params.Offset5, params.GrdStationNum, params.ism)
            G = nx.DiGraph()
            G.add_nodes_from(range(params.graph_node_num))
            ## 1. Inter-satellite links
            for e in edge_list:
                if (e[0] == e[1]) :
                    continue
                if random.random() < self.num_failure:
                    G.add_edge(e[0], e[1], capacity=0)
                else:
                    G.add_edge(e[0], e[1], capacity=params.isl_cap)
                # G.add_edge(e[1], e[0], capacity=params.isl_cap)
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
            
            # print(G.number_of_nodes(), G.number_of_edges())
            # print(len(G))
            return G
        
        else:
            G = nx.DiGraph()
            G.add_nodes_from(range(66*2))
            for e in edge_list:
                G.add_edge(e[0], e[1], capacity=25)
            for i in range(66):
                G.add_edge(i, i+66, capacity=100)
                G.add_edge(i+66, i, capacity=100)
            return G


    def extract_sol_mat(self, action):
        """return sparse solution matrix.
        Solution matrix is of dimension num_of_demand x num_of_edge.
        The i, j entry represents the traffic flow from demand i on edge j.
        """

        # 3D sparse matrix to represent which path, which demand, which edge
        sol_mat_index = torch.stack([
            self.p2e[0] % self.num_path,
            torch.div(self.p2e[0], self.num_path, rounding_mode='floor'),
            self.p2e[1]])

        # merge allocation from different paths of the same demand
        sol_mat = torch.sparse_coo_tensor(
            sol_mat_index,
            action[self.p2e[0]],
            (self.num_path,
                self.num_path_node//self.num_path,
                self.num_edge_node))
        sol_mat = torch.sparse.sum(sol_mat, [0])

        return sol_mat
