# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from typing import Tuple

import torch
import torch.nn as nn

from rlmeta.core.model import RemotableModel
from rlmeta.core.types import NestedTensor


class DQNModel(RemotableModel):

    @abc.abstractmethod
    def forward(self, obs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward function for PPO model.

        Args:
            obs: A torch.Tensor for observation.

        Returns:
            q: The Q(s, a) value for each action in the current state.
        """

    @abc.abstractmethod
    def act(self, obs: torch.Tensor, eps: torch.Tensor, *args,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Act function will be called remotely by the agent.
        This function should upload the input to the device and download the
        output to cpu.

        Args:
            obs: A torch.Tensor for observation.
            eps: A torch.Tensor for eps value in epsilon-greedy policy.

        Returns:
            action: The final action selected by the model.
        """

    @abc.abstractmethod
    def td_error(self, batch: NestedTensor,
                 gamma: torch.Tensor) -> torch.Tensor:
        """
        """

    @abc.abstractmethod
    def compute_priority(self, batch: NestedTensor) -> torch.Tensor:
        """
        """

    @abc.abstractmethod
    def sync_target_net(self) -> None:
        """
        """
