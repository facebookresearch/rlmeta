# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn

import rlmeta.core.remote as remote

from rlmeta.agents.ppo import PPOModel
from rlmeta.models.actor_critic import DiscreteActorCriticHead
from rlmeta.models.atari import NatureCNNBackbone, ImpalaCNNBackbone


class AtariPPOModel(PPOModel):

    def __init__(self, num_actions: int, network: str = "nature") -> None:
        super().__init__()
        self._num_actions = num_actions
        self._network = network.lower()

        if self._network == "nature":
            self._backbone = NatureCNNBackbone()
            self._head = DiscreteActorCriticHead(self._backbone.output_size,
                                                 [512], num_actions)
        elif self._network == "impala":
            self._backbone = ImpalaCNNBackbone()
            self._head = DiscreteActorCriticHead(self._backbone.output_size,
                                                 [256], num_actions)
        else:
            assert False, "Unsupported network."

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        h = self._backbone(x)
        logpi, v = self._head(h)
        return logpi, v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logpi, v = self.forward(obs)
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(deterministic_policy, greedy_action,
                                 sample_action)
            logpi = logpi.gather(dim=-1, index=action)

        return action, logpi, v
