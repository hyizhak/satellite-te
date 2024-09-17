import math
import numpy as np
import os

import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
from dgl.nn import EdgeGATConv

from .. import AssetManager
from .TopoGNN import TopoGNN
from .AlloGNN import AlloGNN
from .utils import weight_initialization, print_


class SaTEActor(nn.Module):

    def __init__(
            self, sate_env, topo_gnn, layers, decoder_type,
            train_id, device,
            std=1, log_std_min=-10.0, log_std_max=10.0):
        """Initialize SaTE actor.

        Args:
            sate_env: SaTE environment
            num_layer: num of layers in flowGNN
            model_dir: model save directory
            model_save: whether to save the model
            device: device id
            std: std value, -1 if apply neuro networks for std
            log_std_min: lower bound for log std
            log_std_max: upper bound for log std
        """

        super(SaTEActor, self).__init__()

        # sate environment
        self.env = sate_env
        
        self.topo_gnn = topo_gnn

        self.num_path = self.env.num_path
        # self.num_path_node = self.env.num_path_node

        self.train_id = train_id

        # init TopoGNN & AlloGNN
        self.device = device
        self.TopoGNN = TopoGNN(self.env, topo_gnn, layers).to(self.device)
        # self.FlowGNN = FlowGNN(self.env, num_layer).to(self.device)
        problem_G_sample = self.env.obs["problem"]
        in_sizes = {'flow': 1,
                    'path': 1,
                    'link': 16}
        
        self.AlloGNN = AlloGNN(self.env, in_sizes=in_sizes , hidden_size=128, 
                               out_sizes={'path':1}, num_heads=4, decoder=decoder_type,
                               canonical_etypes=problem_G_sample.canonical_etypes).to(self.device)

        # init COMA policy
        self.std = std
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.mean_linear = nn.Linear(
            self.num_path,
            self.num_path).to(self.device)
        # apply neuro networks for std
        if std < 0:
            self.log_std_linear = nn.Linear(
                self.num_path,
                self.num_path).to(self.device)

        # # get model fname
        # self.model_fname = self.model_full_fname(
        #     model_dir, self.env.constellation, num_layer, std)
        # self.model_save = model_save
        # # load model
        # self.load_model()

    def model_full_fname(self, create_dir=False):
        """Return full name of the ML model."""

        return os.path.join(
            AssetManager.model_dir(self.env.work_dir, create_dir=create_dir),
            f'{self.train_id}.pt'
        )

    def init_model(self):
        """Initialize model parameters."""
        logging.info('Initializing spaceTE model')
        self.apply(weight_initialization)

    def load_model(self, quantized, model_path=None):
        """Load from model fname."""
        mpath = self.model_path(True) if model_path is None else model_path
        if quantized :
            logging.info(f'Loading spaceTE model from {mpath}')
            self.load_state_dict(torch.load(mpath, map_location=self.device))
        else:
            logging.info(f'Loading spaceTE model from {mpath}')
            self.load_state_dict(torch.load(mpath, map_location=self.device))

    def save_model(self, losses, epoch):
        """Save from model fname."""
        mpath = self.model_path(True)
        with open(mpath.replace('.pt', '.trainings'), 'w') as f:
            f.write('kl_divergence, total flow, panelty, loss\n')
            for loss in losses:
                f.write(f"{','.join(map(str, loss))}\n")
        mpath = mpath.replace('.pt', f'_{epoch}.pt')
        logging.info(f'Saving spaceTE model to {mpath}')
        torch.save(self.state_dict(), mpath)


    def model_path(self, create_dir=False):
        """Return full name of the ML model."""
        mdir = os.path.join(AssetManager.model_dir(self.env.work_dir, create_dir=create_dir),
            f'{self.train_id}')
        os.makedirs(mdir, exist_ok=True)
        return os.path.join(mdir, 'epoch.pt')

    def forward(self, feature, need_topo=False):
        """Return mean of normal distribution after forward propagation

        Args:
            features: input features including capacity and demands
        """
        x = self.TopoGNN(feature["topo"], feature["capacity"])

        if need_topo:
            return x
        
        feature['problem'].nodes['link'].data['x'] = x

        x = self.AlloGNN(feature['problem'], self.env.edge_index_values.unsqueeze(1))
        x = x.reshape(
            self.env.num_path_node // self.num_path, -1
        )
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

    def evaluate(self, obs, deterministic=False, need_topo=False):
        """Return raw action before softmax split ratio.

        Args:
            obs: input features including capacity and demands
            deterministic: whether to have deterministic action
        """

        # feature = obs.reshape(-1, 1)
        if need_topo:
            mean = self.forward(obs, need_topo=True)
        else:
            mean, std = self.forward(obs)
            
        mean = mean

        # test mode
        if deterministic:
            distribution = None
            raw_action = mean
            log_probability = None
        # train mode
        else:
            # use normal distribution for action
            distribution = Normal(mean, std)
            sample = distribution.rsample()
            raw_action = sample
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
