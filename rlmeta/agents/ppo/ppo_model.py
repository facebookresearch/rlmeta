# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch
import torch.nn as nn

from typing import Tuple

from rlmeta.core.model import RemotableModel


class PPOModel(RemotableModel):
    @abc.abstractmethod
    def forward(self, obs: torch.Tensor, *args,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for PPO model.

        Args:
            obs: A torch.Tensor for observation.

        Returns:
            A tuple for pytorch tensor contains [logpi, v].
            logpi: The log probility for each action.
            v: The value of the current state.
        """

    @abc.abstractmethod
    def act(self, obs: torch.Tensor, deterministic_policy: torch.Tensor, *args,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Act function will be called remotely by the agent.
        This function should upload the input to the device and download the
        output to cpu.

        Args:
            obs: A torch.Tensor for observation.
            deterministic_policy: A torch.Tensor for whether to use
              deterministic_policy.

        Returns:
            A tuple for pytorch tensor contains [action, logpi, v].
            action: The final action selected by the model.
            logpi: The log probility for each action.
            v: The value of the current state.
        """
