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
from rlmeta.agents.ppo import PPORNDModel
from rlmeta.core.rescalers import MomentsRescaler
from rlmeta.core.types import NestedTensor


class AtariPPORNDModel(PPORNDModel):

    def __init__(self, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim

        self.ppo_net = AtariBackbone()
        self.tgt_net = AtariBackbone()
        self.prd_net = AtariBackbone()

        self.linear_p = nn.Linear(self.ppo_net.output_dim, self.action_dim)
        self.linear_ext_v = nn.Linear(self.ppo_net.output_dim, 1)
        self.linear_int_v = nn.Linear(self.ppo_net.output_dim, 1)

    def forward(
            self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        h = self.ppo_net(x)
        p = self.linear_p(h)
        logpi = F.log_softmax(p, dim=-1)
        ext_v = self.linear_ext_v(h)
        int_v = self.linear_int_v(h)

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
            tgt = self.tgt_net(x)
        prd = self.prd_net(x)
        err = (prd - tgt).square().mean(-1, keepdim=True)
        return err
