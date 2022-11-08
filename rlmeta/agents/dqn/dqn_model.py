# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rlmeta.core.model import RemotableModel
from rlmeta.core.types import NestedTensor


class DQNModel(RemotableModel):

    @abc.abstractmethod
    def forward(self, observation: torch.Tensor, *args,
                **kwargs) -> torch.Tensor:
        """
        Forward function for DQN model.

        Args:
            observation: A torch.Tensor for observation.

        Returns:
            q: The Q(s, a) value for each action in the current state.
        """

    @abc.abstractmethod
    def q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Q function for DQN model.

        Args:
            s: A torch.Tensor for observation.
            a: A torch.Tensor for action.

        Returns:
            q: The Q(s, a) value for each action in the current state.
        """

    @abc.abstractmethod
    def act(self, observation: NestedTensor, eps: torch.Tensor, *args,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Act function will be called remotely by the agent.
        This function should upload the input to the device and download the
        output to cpu.

        Args:
            observation: A torch.Tensor for observation.
            eps: A torch.Tensor for eps value in epsilon-greedy policy.

        Returns:
            action: The final action selected by the model.
            q: The Q(s, a) value of the current state and action.
            v: The value estimation of current state by max(Q(s, a)).
        """

    @abc.abstractmethod
    def compute_priority(self, observation: NestedTensor, action: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
        """
        """

    @abc.abstractmethod
    def sync_target_net(self) -> None:
        """
        """

    def td_error(self, observation: NestedTensor, action: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        return target - self.q(observation, action)
