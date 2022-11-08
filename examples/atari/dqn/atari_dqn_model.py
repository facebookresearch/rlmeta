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
from rlmeta.core.types import NestedTensor


class AtariDQNNet(nn.Module):

    def __init__(self, action_dim: int, dueling_dqn: bool = True) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._dueling_dqn = dueling_dqn

        self._backbone = AtariBackbone()
        self._linear_a = nn.Linear(self._backbone.output_dim, action_dim)
        self._linear_v = nn.Linear(self._backbone.output_dim,
                                   1) if dueling_dqn else None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = observation.float() / 255.0
        h = self._backbone(x)
        a = self._linear_a(h)
        if self._dueling_dqn:
            v = self._linear_v(h)
            return v + a - a.mean(-1, keepdim=True)
        else:
            return a


class AtariDQNModel(DQNModel):

    def __init__(self,
                 action_dim: int,
                 dueling_dqn: bool = True,
                 double_dqn: bool = False) -> None:
        super().__init__()

        self._action_dim = action_dim
        self._dueling_dqn = dueling_dqn
        self._double_dqn = double_dqn

        # Bootstrapping with online network when double_dqn = False.
        # https://arxiv.org/pdf/2209.07550.pdf
        self._online_net = AtariDQNNet(action_dim, dueling_dqn)
        self._target_net = copy.deepcopy(
            self._online_net) if double_dqn else None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self._online_net(observation)

    def q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q = self._online_net(s)
        q = q.gather(dim=-1, index=a)
        return q

    @remote.remote_method(batch_size=128)
    def act(self, observation: torch.Tensor,
            eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q = self._online_net(observation)  # size = (batch_size, action_dim)
            _, action_dim = q.size()
            greedy_action = q.argmax(-1, keepdim=True)

            pi = torch.ones_like(q) * (eps / (action_dim - 1))
            pi.scatter_(dim=-1, index=greedy_action, src=1.0 - eps)
            action = pi.multinomial(1, replacement=True)
            v = self._value(observation, q)
            q = q.gather(dim=-1, index=action)

        return action, q, v

    @remote.remote_method(batch_size=None)
    def compute_priority(self, observation: NestedTensor, action: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
        td_err = self.td_error(observation, action, target)
        return td_err.squeeze(-1).abs()

    def sync_target_net(self) -> None:
        if self._target_net is not None:
            self._target_net.load_state_dict(self._online_net.state_dict())

    def _value(self,
               observation: torch.Tensor,
               q: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q is None:
            q = self._online_net(observation)
        if not self._double_dqn:
            v = q.max(-1, keepdim=True)[0]
        else:
            a = q.argmax(-1, keepdim=True)
            q = self._target_net(observation)
            v = q.gather(dim=-1, index=a)
        return v
