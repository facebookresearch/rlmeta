# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from typing import Tuple

from rlmeta.agents.dqn.dqn_model import DQNModel
from rlmeta.core.types import NestedTensor


class AtariDQNNet(nn.Module):
    def __init__(self, dueling_dqn: bool = True) -> None:
        super().__init__()
        self.dueling_dqn = dueling_dqn

        layers = []
        layers.append(nn.Conv2d(4, 32, kernel_size=8, stride=4))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(3136, 512))
        layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.linear_a = nn.Linear(512, 6)
        self.linear_v = nn.Linear(512, 1) if dueling_dqn else None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0
        h = self.backbone(x)
        a = self.linear_a(h)
        if self.dueling_dqn:
            v = self.linear_v(h)
            return v + a - a.mean(-1, keepdim=True)
        else:
            return a


class AtariDQNModel(DQNModel):
    def __init__(self,
                 double_dqn: bool = True,
                 dueling_dqn: bool = True) -> None:
        super().__init__()
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

        self.online_net = AtariDQNNet()
        self.target_net = copy.deepcopy(
            self.online_net) if double_dqn else None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.online_net(obs)

    @remote.remote_method(batch_size=128)
    def act(self, obs: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device

        with torch.no_grad():
            x = obs.to(device)
            eps = eps.to(device)  # size = (batch_size, 1)
            q = self.forward(x)  # size = (batch_size, action_dim)
            _, action_dim = q.size()
            greedy_action = q.argmax(-1, keepdim=True)

            pi = torch.ones_like(q) * (eps / (action_dim - 1))
            v = 1.0 - eps  # size = (batch_size, 1)
            pi.scatter_(dim=-1, index=greedy_action, src=v)
            action = pi.multinomial(1, replacement=True)

            return action.cpu()

    def td_error(self, batch: NestedTensor,
                 gamma: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        reward = batch["reward"].to(device)
        next_obs = batch["next_obs"].to(device)
        done = batch["done"].to(device)
        gamma = gamma.to(device)

        q = self.online_net(obs)
        q = q.gather(dim=-1, index=action)

        with torch.no_grad():
            q_next = self.online_net(next_obs)
            if self.double_dqn:
                a_next = q_next.argmax(-1, keepdim=True)
                q_next = self.target_net(next_obs)
                q_next = q_next.gather(dim=-1, index=a_next)
            else:
                q_next = q_next.max(-1, keepdim=True)[0]
            y = torch.where(done, reward, reward + gamma * q_next)

        return (y - q).squeeze(-1)

    @remote.remote_method(batch_size=None)
    def compute_priority(self, batch: NestedTensor,
                         gamma: torch.Tensor) -> torch.Tensor:
        err = self.td_error(batch, gamma)
        return err.abs().cpu()

    def double_dqn_sync(self) -> None:
        if self.double_dqn:
            self.target_net.load_state_dict(self.online_net.state_dict())
