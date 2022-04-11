# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from examples.atari.backbone import AtariBackbone
from rlmeta.agents.dqn.dqn_model import DQNModel
from rlmeta.core.rescalers import SqrtRescaler
from rlmeta.core.types import NestedTensor


class AtariDQNNet(nn.Module):

    def __init__(self, action_dim: int, dueling_dqn: bool = True) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.dueling_dqn = dueling_dqn

        self.backbone = AtariBackbone()
        self.linear_a = nn.Linear(self.backbone.output_dim, self.action_dim)
        self.linear_v = nn.Linear(self.backbone.output_dim,
                                  1) if dueling_dqn else None

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
                 action_dim: int,
                 double_dqn: bool = True,
                 dueling_dqn: bool = True,
                 reward_rescaling: bool = True) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.reward_rescaling = reward_rescaling

        self.online_net = AtariDQNNet(self.action_dim)
        self.target_net = copy.deepcopy(self.online_net)

        if self.reward_rescaling:
            self.rescaler = SqrtRescaler()

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
            if self.double_dqn:
                q_next = self.online_net(next_obs)
                a_next = q_next.argmax(-1, keepdim=True)
                q_next = self.target_net(next_obs)
                q_next = q_next.gather(dim=-1, index=a_next)
            else:
                q_next = self.target_net(next_obs)
                q_next = q_next.max(-1, keepdim=True)[0]
            if self.reward_rescaling:
                q_next = self.rescaler.recover(q_next)
            y = torch.where(done, reward, reward + gamma * q_next)
            if self.reward_rescaling:
                y = self.rescaler.rescale(y)

        return (y - q).squeeze(-1)

    @remote.remote_method(batch_size=None)
    def compute_priority(self, batch: NestedTensor,
                         gamma: torch.Tensor) -> torch.Tensor:
        err = self.td_error(batch, gamma)
        return err.abs().cpu()

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
