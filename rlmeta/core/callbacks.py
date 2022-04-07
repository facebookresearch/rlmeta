# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from rlmeta.core.types import Action, TimeStep


class EpisodeCallbacks:
    """Callbacks class for custom episode metrics.
    
    Similar as the DefaultCallbacks in RLLib.
    https://github.com/ray-project/ray/blob/master/rllib/agents/callbacks.py
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
