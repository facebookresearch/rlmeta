# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from typing import Any, NamedTuple, Optional, Union

Tensor = Union[np.ndarray, torch.Tensor]

# NestedTensor is adapted from Acme's NestedTensor
# https://github.com/deepmind/acme/blob/df961057bcd2e1436d5f894ebced62d694225034/acme/types.py#L23
#
# It was released under the Apache License, Version 2.0 (the "License"),
# available at:
# http://www.apache.org/licenses/LICENSE-2.0
NestedTensor = Any


# TimeStep is Inspired from dm_env's TimeStep:
# https://github.com/deepmind/dm_env/blob/abee135a07cc8e684173586dc8a20e696bbd40fb/dm_env/_environment.py#L25
#
# It was released under the Apache License, Version 2.0 (the "License"),
# available at:
# http://www.apache.org/licenses/LICENSE-2.0
class TimeStep(NamedTuple):
    observation: Any
    reward: Optional[float] = None
    done: bool = False
    info: Optional[Any] = None


class Action(NamedTuple):
    action: Any
    info: Optional[Any] = None
