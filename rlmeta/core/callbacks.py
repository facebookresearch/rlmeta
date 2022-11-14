# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from rlmeta.core.types import Action, TimeStep

# The EpisodeCallbacks class is adapted from RLLib's DefaultCallbacks
# https://github.com/ray-project/ray/blob/f9173a189023ccf4b4b09cf1533c628da13d000b/rllib/algorithms/callbacks.py#L37
#
# It was released under the Apache License, Version 2.0 (the "License"),
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


class EpisodeCallbacks:
    """Callbacks class for custom episode metrics.
    """

    def __init__(self) -> None:
        self._custom_metrics = {}

    @property
    def custom_metrics(self) -> Dict[str, Any]:
        return self._custom_metrics

    @custom_metrics.setter
    def custom_metrics(self, metrics: Dict[str, Any]) -> None:
        self._custom_metrics = metrics

    def reset(self) -> None:
        self._custom_metrics.clear()

    def on_episode_start(self, index: int) -> None:
        pass

    def on_episode_init(self, index: int, timestep: TimeStep) -> None:
        pass

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        pass

    def on_episode_end(self, index: int) -> None:
        pass
