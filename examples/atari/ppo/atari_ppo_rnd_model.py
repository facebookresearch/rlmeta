# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from rlmeta.agents.ppo import PPORNDModel
from rlmeta.core.types import NestedTensor
from rlmeta.models.actor_critic import DiscreteActorCriticRNDHead
from rlmeta.models.atari import NatureCNNBackbone, ImpalaCNNBackbone


class AtariPPORNDModel(PPORNDModel):

    def __init__(self, num_actions: int, network: str = "nature") -> None:
        super().__init__()

        self._num_actions = num_actions
        self._network = network.lower()

        if self._network == "nature":
            self._ppo_net = NatureCNNBackbone()
            self._tgt_net = NatureCNNBackbone()
            self._prd_net = NatureCNNBackbone()
            self._head = DiscreteActorCriticRNDHead(self._ppo_net.output_size,
                                                    [512], num_actions)

        elif self._network == "impala":
            self._ppo_net = ImpalaCNNBackbone()
            self._tgt_net = ImpalaCNNBackbone()
            self._prd_net = ImpalaCNNBackbone()
            self._head = DiscreteActorCriticRNDHead(self._ppo_net.output_size,
                                                    [256], num_actions)
        else:
            assert False, "Unsupported network."

    def forward(
            self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        h = self._ppo_net(x)
        logpi, ext_v, int_v = self._head(h)
        return logpi, ext_v, int_v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logpi, ext_v, int_v = self.forward(obs)
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(deterministic_policy, greedy_action,
                                 sample_action)
            logpi = logpi.gather(dim=-1, index=action)

        return action, logpi, ext_v, int_v

    @remote.remote_method(batch_size=None)
    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._rnd_error(obs)

    def rnd_loss(self, obs: torch.Tensor) -> torch.Tensor:
        return self._rnd_error(obs).mean() * 0.5

    def _rnd_error(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0
        with torch.no_grad():
            tgt = self._tgt_net(x)
        prd = self._prd_net(x)
        err = (prd - tgt).square().mean(-1, keepdim=True)
        return err
