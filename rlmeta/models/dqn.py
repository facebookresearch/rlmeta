# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
import torch.nn as nn

from rlmeta.models.utils import MLP


class DQNHead(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: Sequence[int],
                 num_actions: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._num_actions = num_actions
        self._mlp = MLP(input_size, [*hidden_sizes, num_actions])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class DuelingDQNHead(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: Sequence[int],
                 num_actions: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._num_actions = num_actions
        self._mlp_a = MLP(input_size, [*hidden_sizes, num_actions])
        self._mlp_v = MLP(input_size, [*hidden_sizes, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self._mlp_a(x)
        v = self._mlp_v(x)
        return v + a - a.mean(dim=-1, keepdim=True)
