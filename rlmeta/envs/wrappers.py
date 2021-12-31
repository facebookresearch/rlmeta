# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env


# Simiar as TimeLimit in OpenAI baselines.
# https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
class TimeLimitWrapper(Env):

    def __init__(self, env: Env, max_episode_steps: int) -> None:
        super().__init__()
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def reset(self, *args, **kwargs) -> TimeStep:
        self._elapsed_steps = 0
        return self._env.reset(*args, **kwargs)

    def step(self, action: Action) -> TimeStep:
        obs, reward, done, info = self._env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        return TimeStep(obs, reward, done, info)

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)
