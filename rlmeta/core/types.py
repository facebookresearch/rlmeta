# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from typing import Any, NamedTuple, Optional, Union

Tensor = Union[np.ndarray, torch.Tensor]
NestedTensor = Any


# Inspired from dm_env
# https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py#L25
class TimeStep(NamedTuple):
    observation: Any
    reward: Optional[float] = None
    done: bool = False
    info: Optional[Any] = None


class Action(NamedTuple):
    action: Any
    info: Optional[Any] = None
