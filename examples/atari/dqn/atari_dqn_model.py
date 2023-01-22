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

from rlmeta.agents.dqn import DQNModel
from rlmeta.core.types import NestedTensor
from rlmeta.models.atari import NatureCNNBackbone, ImpalaCNNBackbone
from rlmeta.models.dqn import DQNHead, DuelingDQNHead


class AtariDQNNet(nn.Module):

    def __init__(self,
                 num_actions: int,
                 network="nature",
                 dueling_dqn: bool = True,
                 spectral_norm: bool = True) -> None:
        super().__init__()
        self._num_actions = num_actions
        self._network = network.lower()
        self._dueling_dqn = dueling_dqn
        self._spectral_norm = spectral_norm

        head_cls = DuelingDQNHead if dueling_dqn else DQNHead
        if self._network == "nature":
            self._backbone = NatureCNNBackbone()
            self._head = head_cls(self._backbone.output_size, [512],
                                  num_actions)
        elif self._network == "impala":
            self._backbone = ImpalaCNNBackbone()
            self._head = head_cls(self._backbone.output_size, [256],
                                  num_actions)
        else:
            assert False, "Unsupported network."

    def init_model(self) -> None:
        if self._spectral_norm:
            # Apply SN[-2] in https://arxiv.org/pdf/2105.05246.pdf
            nn.utils.parametrizations.spectral_norm(
                self._head._mlp_a._layers[-3])
            nn.utils.parametrizations.spectral_norm(
                self._head._mlp_v._layers[-3])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = observation.float() / 255.0
        h = self._backbone(x)
        a = self._head(h)
        return a


class AtariDQNModel(DQNModel):

    def __init__(self,
                 num_actions: int,
                 network: str = "nature",
                 dueling_dqn: bool = True,
                 spectral_norm: bool = True,
                 double_dqn: bool = False) -> None:
        super().__init__()

        self._num_actions = num_actions
        self._network = network.lower()
        self._dueling_dqn = dueling_dqn
        self._spectral_norm = spectral_norm
        self._double_dqn = double_dqn

        # Bootstrapping with online network when double_dqn = False.
        # https://arxiv.org/pdf/2209.07550.pdf
        self._online_net = AtariDQNNet(num_actions,
                                       network=network,
                                       dueling_dqn=dueling_dqn,
                                       spectral_norm=spectral_norm)
        self._target_net = copy.deepcopy(
            self._online_net) if double_dqn else None

    def init_model(self) -> None:
        self._online_net.init_model()
        if self._target_net is not None:
            self._target_net.init_model()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self._online_net(observation)

    def q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q = self._online_net(s)
        q = q.gather(dim=-1, index=a)
        return q

    @remote.remote_method(batch_size=256)
    def act(self, observation: torch.Tensor,
            eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q = self._online_net(observation)  # size = (batch_size, action_dim)
            _, action_dim = q.size()
            greedy_action = q.argmax(-1, keepdim=True)

            pi = torch.ones_like(q) * (eps / action_dim)
            pi.scatter_(dim=-1,
                        index=greedy_action,
                        src=1.0 - eps * (action_dim - 1) / action_dim)
            action = pi.multinomial(1)
            v = self._value(observation, q)
            q = q.gather(dim=-1, index=action)

        return action, q, v

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
