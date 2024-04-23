import pickle
import time
import json
import sys
import os
from tqdm import tqdm
from networkx.readwrite import json_graph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dytop_actor import DyToPActor
from .dytop_env import DyToPEnv
from .utils import print_


class DyToP():
    def __init__(self, dytop_env, dytop_actor, lr, early_stop):
        """Initialize DyToP model.

        Args:
            dytop_env: DyToP environment
            dytop_actor: DyToP actor
            lr: learning rate
            early_stop: whether to early stop
        """

        self.env = dytop_env
        self.actor = dytop_actor

        # init optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # early stop when val result no longer changes
        self.early_stop = early_stop
        if self.early_stop:
            self.val_reward = []

    def train(self, num_epoch, batch_size, num_sample, save_model=False):
        """Train DyToP model.

        Args:
            num_epoch: number of training epoch
            batch_size: batch size
            num_sample: number of samples in COMA reward
        """

        for epoch in range(num_epoch):

            self.env.reset('train')

            ids = range(0, self.env.idx_stop)
            loop_obj = tqdm(
                [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)],
                desc=f"Training epoch {epoch+1}/{num_epoch}: ", position=0)

            for idx in loop_obj:
                loss = 0
                for _ in idx:
                    torch.cuda.empty_cache()

                    # get observation
                    obs = self.env.get_obs()
                    # get action
                    raw_action, log_probability = self.actor.evaluate(obs)
                    # get reward
                    reward, info = self.env.step(
                        raw_action, num_sample=num_sample)
                    loss += -(log_probability*reward).mean()

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                # break

            # early stop
            if self.early_stop:
                self.val()
                if len(self.val_reward) > 20 and abs(
                        sum(self.val_reward[-20:-10])/10
                        - sum(self.val_reward[-10:])/10) < 0.0001:
                    break
        if save_model:
            self.save_model()

    def val(self):
        """Validating DyToP model."""

        self.actor.eval()
        self.env.reset('val')

        rewards = 0
        for idx in range(self.env.idx_start, self.env.idx_stop):

            # get observation
            problem_dict = self.env.render()
            obs = self.env.get_obs()
            # get action
            raw_action = self.actor.act(obs)
            # get reward
            reward, info = self.env.step(raw_action)
            # show satisfied demand instead of total flow
            rewards += reward.item()/problem_dict['total_demand']\
                if self.env.obj == 'total_flow' else reward.item()
        self.val_reward.append(
            rewards/(self.env.idx_stop - self.env.idx_start))

    def test(self, num_admm_step, output_header, output_placeholder, output_csv):
        """Test DyToP model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.reset('test')

        with open(output_csv, "a") as resultf:

            runtime_list, obj_list = [], []
            loop_obj = tqdm(
                range(0, self.env.idx_stop),
                desc="Testing: ")

            for idx in loop_obj:

                # get observation
                problem_dict = self.env.render()
                obs = self.env.get_obs()
                # get action
                start_time = time.time()
                raw_action = self.actor.act(obs)
                runtime = time.time() - start_time
                # get reward
                reward, info = self.env.step(
                    raw_action, num_admm_step=num_admm_step)
                # add runtime in transforming, ADMM, rounding
                runtime += info['runtime']
                runtime_list.append(runtime)
                # show satisfied demand instead of total flow
                obj_list.append(
                    reward.item()/problem_dict['total_demand']
                    if self.env.obj == 'total_flow' else reward.item())

                # display avg runtime, obj
                loop_obj.set_postfix({
                    'runtime': '%.4f' % (sum(runtime_list)/len(runtime_list)),
                    'obj': '%.4f' % (sum(obj_list)/len(obj_list)),
                    })

                assert problem_dict['obj'] == 'total_flow'

                result_line = output_placeholder.format(
                    problem_dict['topo_idx'],
                    problem_dict['tm_idx'],
                    problem_dict['total_demand'],
                    reward,
                    reward / problem_dict['total_demand'],
                    runtime)

                print(result_line, file=resultf)

    def save_model(self):
        self.actor.save_model()

    def load_model(self, quantized, compiled):
        self.actor.load_model(quantized)
        if compiled:
            print("JIT-compiling")
            self.actor = torch.compile(self.actor, mode="max-autotune")
