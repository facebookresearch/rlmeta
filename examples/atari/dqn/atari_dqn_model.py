# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote
import rlmeta.utils.nested_utils as nested_utils

from examples.atari.backbone import AtariBackbone
from rlmeta.agents.dqn import DQNModel
# from rlmeta.core.rescalers import SqrtRescaler
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
                 dueling_dqn: bool = True) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

        self.online_net = AtariDQNNet(self.action_dim)
        self.target_net = copy.deepcopy(self.online_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.online_net(obs)

    @remote.remote_method(batch_size=128)
    def act(self, obs: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device

        with torch.no_grad():
            x = obs.to(device)
            eps = eps.to(device)  # size = (batch_size, 1)
            q = self.online_net(x)  # size = (batch_size, action_dim)
            _, action_dim = q.size()
            greedy_action = q.argmax(-1, keepdim=True)

            pi = torch.ones_like(q) * (eps / (action_dim - 1))
            pi.scatter_(dim=-1, index=greedy_action, src=1.0 - eps)
            action = pi.multinomial(1, replacement=True)
            v = self._value(x, q)

            return action.cpu(), v.cpu()

    def td_error(self, batch: NestedTensor) -> torch.Tensor:
        obs = batch["obs"]
        action = batch["action"]
        target = batch["target"]
        q = self.online_net(obs)
        q = q.gather(dim=-1, index=action)
        return (target - q).squeeze(-1)

    @remote.remote_method(batch_size=None)
    def compute_priority(self, batch: NestedTensor) -> torch.Tensor:
        device = next(self.parameters()).device
        batch = nested_utils.map_nested(lambda x: x.to(device), batch)
        err = self.td_error(batch)
        return err.abs().cpu()

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _value(self,
               obs: torch.Tensor,
               q: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.double_dqn:
            if q is None:
                q = self.online_net(obs)
            a = q.argmax(-1, keepdim=True)
            q = self.target_net(obs)
            v = q.gather(dim=-1, index=a)
        else:
            q = self.target_net(obs)
            v = q.max(-1, keepdim=True)[0]
        return v
