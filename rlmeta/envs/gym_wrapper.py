# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import numpy as np

import gym

from gym.wrappers.step_api_compatibility import StepAPICompatibility

import rlmeta.utils.data_utils as data_utils

from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.envs.env import Env


class GymWrapper(Env):

    def __init__(self,
                 env: gym.Env,
                 observation_fn: Optional[Callable[..., Tensor]] = None,
                 old_step_api: bool = False) -> None:
        super(GymWrapper, self).__init__()

        self._env = StepAPICompatibility(
            env, output_truncation_bool=True) if old_step_api else env
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._reward_range = self._env.reward_range
        self._metadata = self._env.metadata
        self._old_step_api = old_step_api

        if observation_fn is not None:
            self._observation_fn = observation_fn
        else:
            self._observation_fn = data_utils.to_torch

    @property
    def env(self):
        return self._env

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_range(self):
        return self._reward_range

    @property
    def metadata(self):
        return self._metadata

    def reset(self, *args, seed: Optional[int] = None, **kwargs) -> TimeStep:
        # TODO: Clean up this function when most envs are fully migrated to the
        # new OpenAI Gym API.
        if self._old_step_api:
            if seed is not None:
                self._env.seed(seed)
            obs = self._env.reset(*args, **kwargs)
            info = None
        else:
            obs, info = self._env.reset(*args, seed=seed, **kwargs)
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)
        obs = self._observation_fn(obs)
        return TimeStep(obs, info=info)

    def step(self, action: Action) -> TimeStep:
        act = action.action
        if not isinstance(act, int):
            act = act.item()
        obs, reward, terminated, truncated, info = self._env.step(act)
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)
        obs = self._observation_fn(obs)
        return TimeStep(obs, reward, terminated, truncated, info)

    def close(self) -> None:
        self._env.close()
