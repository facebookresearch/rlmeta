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

    def __init__(self,
                 action_dim: int,
                 observation_normalization: bool = False) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.observation_normalization = observation_normalization
        if self.observation_normalization:
            self.obs_rescaler = MomentsRescaler(size=(4, 84, 84))

        self.policy_net = AtariBackbone()
        self.target_net = AtariBackbone()
        self.predict_net = AtariBackbone()
        self.linear_p = nn.Linear(self.policy_net.output_dim, self.action_dim)
        self.linear_ext_v = nn.Linear(self.policy_net.output_dim, 1)
        self.linear_int_v = nn.Linear(self.policy_net.output_dim, 1)

    def forward(
            self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.float() / 255.0
        h = self.policy_net(x)
        p = self.linear_p(h)
        logpi = F.log_softmax(p, dim=-1)
        ext_v = self.linear_ext_v(h)
        int_v = self.linear_int_v(h)

        return logpi, ext_v, int_v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        with torch.no_grad():
            x = obs.to(device)
            d = deterministic_policy.to(device)
            logpi, ext_v, int_v = self.forward(x)

            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), ext_v.cpu(), int_v.cpu()

    @remote.remote_method(batch_size=None)
    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        reward = self._rnd_error(obs.to(device))
        return reward.cpu()

    def rnd_loss(self, obs: torch.Tensor) -> torch.Tensor:
        return self._rnd_error(obs).mean() * 0.5

    def _rnd_error(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0
        if self.observation_normalization:
            self.obs_rescaler.update(x)
            x = self.obs_rescaler.rescale(x)

        with torch.no_grad():
            target = self.target_net(x)
        pred = self.predict_net(x)
        err = (pred - target).square().mean(-1, keepdim=True)

        return err
