# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from examples.atari.backbone import AtariBackbone
from rlmeta.agents.ppo import PPOModel


class AtariPPOModel(PPOModel):

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.backbone = AtariBackbone()
        self.linear_p = nn.Linear(self.backbone.output_dim, self.action_dim)
        self.linear_v = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        h = self.backbone(x)
        p = self.linear_p(h)
        logpi = F.log_softmax(p, dim=-1)
        v = self.linear_v(h)
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
