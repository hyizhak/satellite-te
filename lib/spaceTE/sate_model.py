import pickle
import time
import json
import sys
import os
from tqdm import tqdm
from networkx.readwrite import json_graph
from ot.lp import wasserstein_1d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .sate_actor import SaTEActor
from .sate_env import SaTEEnv
from .utils import print_, smoothing


class SaTE():
    def __init__(self, sate_env, sate_actor, lr, supervised, penalized, loss, early_stop):
        """Initialize SaTE model.

        Args:
            sate_env: SaTE environment
            sate_actor: SaTE actor
            lr: learning rate
            early_stop: whether to early stop
        """

        self.env = sate_env
        self.actor = sate_actor

        # init optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.supervised = supervised
        self.penalized = penalized
        self.loss = loss

        # early stop when val result no longer changes
        self.early_stop = early_stop
        if self.early_stop:
            self.val_reward = []

    def train(self, num_epoch, batch_size, num_sample, save_model=False):
        """Train SaTE model.

        Args:
            num_epoch: number of training epoch
            batch_size: batch size
            num_sample: number of samples in COMA reward
        """

        self.losses = []
        best_loss = float('inf')
        early_stop_count = 0

        for epoch in range(num_epoch):

            self.env.reset('train')

            ids = range(0, self.env.idx_stop)
            loop_obj = tqdm(
                [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)],
                desc=f"Training epoch {epoch+1}/{num_epoch}: ", position=0, mininterval=600)

            for idx in loop_obj:
                satisfied_ratio, total_flow, panelty, loss = 0, 0, 0, 0
                self.actor_optimizer.zero_grad()
                for _ in idx:
                    torch.cuda.empty_cache()

                    # get observation
                    obs = self.env.get_obs()
                    if self.supervised or self.penalized:
                        # get action
                        raw_action, _ = self.actor.evaluate(obs, deterministic=True)
                        # get loss
                        split_loss, batch_loss = self.env.compute_loss(raw_action)
                        satisfied_ratio += split_loss[0]
                        total_flow += split_loss[1]
                        panelty += split_loss[2]
                        loss += batch_loss
                    else:
                        # get action
                        raw_action, log_probability = self.actor.evaluate(obs)
                        # get reward
                        reward, info = self.env.step(
                            raw_action, num_sample=num_sample)
                        loss += -(log_probability*reward).mean()

                loss.backward()
                self.actor_optimizer.step()
                loss_list = [satisfied_ratio.item(), total_flow.item(), panelty.item(), loss.item()]
                    
                self.losses.append([x / len(idx) for x in loss_list])

            self.draw_loss()
            # if epoch > 3 * num_epoch // 4:
            self.save_model(epoch+1)

            # early stop
            if self.early_stop:
                self.val()
                if len(self.val_reward) > 20 and abs(
                        sum(self.val_reward[-20:-10])/10
                        - sum(self.val_reward[-10:])/10) < 0.0001:
                    break

    def val(self):
        """Validating SaTE model."""

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

    def test(self, num_admm_step, output_header, output_placeholder, output_csv, admm_test):
        """Test SaTE model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.reset('test')

        with open(output_csv, "a") as resultf:

            runtime_list, obj_list, test_losses = [], [], []
            loop_obj = tqdm(
                range(0, self.env.idx_stop),
                desc="Testing: ", mininterval=600)

            for idx in loop_obj:

                # get observation
                problem_dict = self.env.render()
                obs = self.env.get_obs()
                # get action
                start_time = time.time()
                raw_action = self.actor.act(obs)
                runtime = time.time() - start_time
                # get reward
                if admm_test:
                    pseudo_action = torch.ones(raw_action.shape).to(raw_action.device)
                    reward, info = self.env.step(
                        pseudo_action, num_admm_step=num_admm_step)
                    # label = self.env.get_label()
                    # reward, info = self.env.step(
                    #     label, num_admm_step=num_admm_step)
                else:
                    try:
                        label = self.env.get_label()
                    except:
                        label = None

                    action = self.env.transform_raw_action(raw_action, prob=True)

                    if label is not None:
                        if self.loss == 'kl_div':
                            test_losses.append(F.kl_div(action.log(), label, reduction='batchmean'))
                        elif self.loss == 'wasserstein':
                            test_losses.append(wasserstein_1d(action.T, label.T).mean())
                        
                    reward, info = self.env.step(
                        raw_action, num_admm_step=num_admm_step)
                # add runtime in transforming, ADMM, rounding
                runtime += info['runtime']
                runtime_list.append(runtime)
                # show satisfied demand instead of total flow
                obj_list.append(
                    reward.item()/problem_dict['total_demand']
                    if self.env.obj.endswith('total_flow') else reward.item())

                # display avg runtime, obj
                # loop_obj.set_postfix({
                #     'runtime': '%.4f' % (sum(runtime_list)/len(runtime_list)),
                #     'obj': '%.4f' % (sum(obj_list)/len(obj_list)),
                #     })

                assert problem_dict['obj'].endswith('total_flow')

                result_line = output_placeholder.format(
                    problem_dict['topo_idx'],
                    problem_dict['tm_idx'],
                    problem_dict['total_demand'],
                    reward,
                    reward / problem_dict['total_demand'],
                    runtime)

                print(result_line, file=resultf)

            runtime_avg = sum(runtime_list) / len(runtime_list)
            obj_avg = sum(obj_list) / len(obj_list)
            print(f'runtime: {runtime_avg:.4f}, obj: {obj_avg:.4f}')
            if len(test_losses) > 0:
                print(f'{self.loss}: {sum(test_losses)/len(test_losses):.4f}')

    def draw_loss(self, smoothing_window=500):
        kl_divergence, total_flow, penalty, loss = zip(*self.losses)
        steps = range(1, len(self.losses) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        # set title to os.path.basename(model_path)
        plt.suptitle(os.path.basename(self.actor.model_path()).replace('.pt', ''))

        if len(steps) > smoothing_window:
            kl_divergence = smoothing(kl_divergence, smoothing_window)
            total_flow = smoothing(total_flow, smoothing_window)
            penalty = smoothing(penalty, smoothing_window)
            loss = smoothing(loss, smoothing_window)
            steps = range(smoothing_window, len(loss) + smoothing_window)

        axs[0, 0].plot(steps, kl_divergence)
        axs[0, 0].set_title('Satisfied Ratio')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].set_ylabel('Satisfied Ratio')
        

        axs[0, 1].plot(steps, total_flow)
        if self.supervised and self.penalized:
            axs[0, 1].set_title(self.loss)
            axs[0, 1].set_ylabel(self.loss)
        else:
            axs[0, 1].set_title('Total Flow')
            axs[0, 1].set_ylabel('Total Flow')
        axs[0, 1].set_xlabel('Steps')

        axs[1, 0].plot(steps, penalty)
        axs[1, 0].set_title('Penalty')
        axs[1, 0].set_xlabel('Steps')
        axs[1, 0].set_ylabel('Penalty')

        axs[1, 1].plot(steps, loss)
        if self.supervised:
            if self.penalized:
                axs[1, 1].set_title('Combined Loss')
                axs[1, 1].set_ylabel('Combined Loss')
            else:
                axs[1, 1].set_title(self.loss)
                axs[1, 1].set_ylabel(self.loss)
        else:
            axs[1, 1].set_title('Penalized Optimization Loss')
            axs[1, 1].set_ylabel('Penalized Optimization Loss')
        axs[1, 1].set_xlabel('Steps')

        plt.tight_layout()
        model_path = self.actor.model_path(create_dir=True)
        model_dir = os.path.dirname(model_path)
        train_log_path = os.path.join(os.path.dirname(model_dir), 'train_logs')
        model_name = os.path.basename(model_path).replace('.pt', '_figure')
        model_folder_path = os.path.join(train_log_path, model_name)
        os.makedirs(model_folder_path, exist_ok=True)
        figure_path = os.path.join(model_folder_path, f'steps-{len(self.losses)}_losses.png')
        plt.savefig(figure_path)

    def save_model(self, epoch):
        self.actor.save_model(self.losses, epoch)

    def load_model(self, quantized, compiled, model_path=None):
        self.actor.load_model(quantized, model_path)
        if compiled:
            print("JIT-compiling")
            self.actor = torch.compile(self.actor, mode="max-autotune")
