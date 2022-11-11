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
# https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/wrappers.py#L3
#
# It is under MIT license
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:


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
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        return TimeStep(obs, reward, terminated, truncated, info)

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)
