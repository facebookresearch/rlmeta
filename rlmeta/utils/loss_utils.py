# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

_NAME_TO_LOSS = {
    "huber": nn.HuberLoss,
    "huber_loss": nn.HuberLoss,
    "huberloss": nn.HuberLoss,
    "l1": nn.L1Loss,
    "l1_loss": nn.L1Loss,
    "l1loss": nn.L1Loss,
    "mse": nn.MSELoss,
    "mse_loss": nn.MSELoss,
    "mseloss": nn.MSELoss,
    "smooth_l1": nn.SmoothL1Loss,
    "smooth_l1_loss": nn.SmoothL1Loss,
    "smoothl1": nn.SmoothL1Loss,
    "smoothl1loss": nn.SmoothL1Loss,
}


def get_loss(name: str, args: Optional[Dict[str, Any]] = None) -> nn.Module:
    loss = _NAME_TO_LOSS[name.lower()]
    return loss(
        reduction="none") if args is None else loss(reduction="none", **args)
