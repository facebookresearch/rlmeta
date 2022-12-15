# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlmeta.models.utils import MLP


class DiscreteActorCriticHead(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: Sequence[int],
                 num_actions: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._num_actions = num_actions
        self._mlp_p = MLP(input_size, [*hidden_sizes, num_actions])
        self._mlp_v = MLP(input_size, [*hidden_sizes, 1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self._mlp_p(x)
        logpi = F.log_softmax(p, dim=-1)
        v = self._mlp_v(x)
        return logpi, v


class DiscreteActorCriticRNDHead(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: Sequence[int],
                 num_actions: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._num_actions = num_actions
        self._mlp_p = MLP(input_size, [*hidden_sizes, num_actions])
        self._mlp_ext_v = MLP(input_size, [*hidden_sizes, 1])
        self._mlp_int_v = MLP(input_size, [*hidden_sizes, 1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self._mlp_p(x)
        logpi = F.log_softmax(p, dim=-1)
        ext_v = self._mlp_ext_v(x)
        int_v = self._mlp_int_v(x)
        return logpi, ext_v, int_v
