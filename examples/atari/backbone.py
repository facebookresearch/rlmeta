# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class AtariBackbone(nn.Module):

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
        layers.append(nn.Linear(3136, 512))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
