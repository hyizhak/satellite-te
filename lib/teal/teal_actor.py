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

from .. import AssetManager
from .FlowGNN import FlowGNN
from .utils import weight_initialization


class TealActor(nn.Module):

    def __init__(
            self, teal_env, layer_num, device,
            train_id:str,
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
        self.env = teal_env
        self.layer_num = layer_num
        
        self.path_num = self.env.num_path
        self.path_node_num = self.env.num_path_node
        
        self.train_id = train_id

        # init FlowGNN
        self.device = device
        self.FlowGNN = FlowGNN(self.env, layer_num).to(self.device)

        # init COMA policy
        self.std = std
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.mean_linear = nn.Linear(
            self.path_num*(self.FlowGNN.num_layer+1),
            self.path_num).to(self.device)
        # apply neuro networks for std
        if std < 0:
            self.log_std_linear = nn.Linear(
                self.path_num*(self.FlowGNN.num_layer+1),
                self.path_num).to(self.device)


    def model_path(self, create_dir=False):
        """Return full name of the ML model."""
        return os.path.join(
            AssetManager.model_dir(self.env.work_dir, self.env.topo_idx, create_dir=create_dir),
            f'{self.train_id}.pt'
        )
    
    def init_model(self):
        """Initialize model parameters."""
        logging.info('Initializing Teal model')
        self.apply(weight_initialization)

    def load_model(self):
        """Load from model fname."""
        mpath = self.model_path(True)
        logging.info(f'Loading Teal model from {mpath}')
        self.load_state_dict(torch.load(mpath, map_location=self.device))

    def save_model(self):
        """Save from model fname."""
        mpath = self.model_path(True)
        logging.info(f'Saving Teal model to {mpath}')
        torch.save(self.state_dict(), mpath)

    def forward(self, feature):
        """Return mean of normal distribution after forward propagation

        Args:
            features: input features including capacity and demands
        """
        x = self.FlowGNN(feature)
        x = x.reshape(
            self.path_node_num//self.path_num,
            self.path_num*(self.FlowGNN.num_layer+1))
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
