# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from typing import Tuple

from rlmeta.agents.ppo.ppo_model import PPOModel


class AtariPPOModel(PPOModel):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

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
        self.linear_p = nn.Linear(512, self.action_dim)
        self.linear_v = nn.Linear(512, 1)

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
        device = next(self.parameters()).device

        with torch.no_grad():
            x = obs.to(device)
            d = deterministic_policy.to(device)
            logpi, v = self.forward(x)

            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), v.cpu()
