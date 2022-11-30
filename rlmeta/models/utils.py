# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_sizes: Sequence[int],
                 activate_last: bool = False) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._activate_final = activate_last

        prev_size = input_size
        last_size = hidden_sizes.pop()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, last_size))
        if activate_last:
            layers.append(nn.ReLU())
        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size

        layers = []
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding="same"))
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding="same"))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._layers(x)
