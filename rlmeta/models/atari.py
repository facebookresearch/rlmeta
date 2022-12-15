# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from rlmeta.models.utils import ResidualBlock


class NatureCNNBackbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(4, 32, kernel_size=8, stride=4))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)

    @property
    def output_size(self) -> int:
        return 3136

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class ImpalaCNNBackbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        layers = []
        layers.append(self._conv_block(4, 16))
        layers.append(self._conv_block(16, 32))
        layers.append(self._conv_block(32, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)

    @property
    def output_size(self) -> int:
        return 3872

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
