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
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NestedTensor = Any

# TimeStep is Inspired from dm_env's TimeStep:
# https://github.com/deepmind/dm_env/blob/abee135a07cc8e684173586dc8a20e696bbd40fb/dm_env/_environment.py#L25
#
# Copyright 2019 The dm_env Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class TimeStep(NamedTuple):
    observation: Any
    reward: Optional[float] = None
    terminated: bool = False
    truncated: bool = False
    info: Optional[Any] = None


class Action(NamedTuple):
    action: Any
    info: Optional[Any] = None
