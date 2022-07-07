# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import gym
import numpy as np
import torch

import rlmeta.envs.atari_wrappers as atari_wrappers
import rlmeta.utils.data_utils as data_utils

from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.wrappers import TimeLimitWrapper


class ImageObservationWrapper(gym.ObservationWrapper):
    """
    Wrap image observation from NHWC order to NCHW order.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        shape = self.observation_space.shape
        dtype = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(shape[2], shape[0],
                                                       shape[1]),
                                                dtype=dtype)

    def observation(self, observation: np.ndarray) -> None:
        return np.ascontiguousarray(np.transpose(observation, axes=(2, 0, 1)))


class GymWrapper(Env):

    def __init__(
            self,
            env: gym.Env,
            observation_fn: Optional[Callable[..., Tensor]] = None) -> None:
        super(GymWrapper, self).__init__()

        self._env = env
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._reward_range = self._env.reward_range
        self._metadata = self._env.metadata

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

    def reset(self, *args, **kwargs) -> TimeStep:
        obs = self._env.reset(*args, **kwargs)
        obs = self._observation_fn(obs)
        return TimeStep(obs, done=False)

    def step(self, action: Action) -> TimeStep:
        act = action.action
        if not isinstance(act, int):
            act = act.item()
        obs, reward, done, info = self._env.step(act)
        obs = self._observation_fn(obs)
        return TimeStep(obs, reward, done, info)

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

class MAGymWrapper(GymWrapper):
    def __init__(
            self,
            env: gym.Env,
            observation_fn: Optional[Callable[..., Tensor]] = None) -> None:
        super(MAGymWrapper, self).__init__(env)
    
    def reset(self, *args, **kwargs) -> TimeStep:
        obs = self._env.reset(*args, **kwargs)
        timestep = {}
        for k,v in obs.items():
            obs[k] = self._observation_fn(obs[k])
            timestep[k] = TimeStep(obs[k], done=False)
        return timestep

    def step(self, action: Action) -> TimeStep: #TODO check action type
        #act = action.action
        #if not isinstance(act, int):
        #    act = act.item()
        act = {}
        for k,v, in action.items():
            act[k] = v.action
            if not isinstance(act[k], int):
                act[k] = act[k].item()

        obs, reward, done, info = self._env.step(act)
        timestep = {}
        for k,v in obs.items():
            obs[k] = self._observation_fn(obs[k])
            timestep[k]=TimeStep(obs[k], reward[k], done[k], info[k])
        return timestep

class AtariWrapperFactory(EnvFactory):

    def __init__(self,
                 env_id: str,
                 max_episode_steps: Optional[int] = None,
                 episode_life: bool = False,
                 clip_rewards: bool = False,
                 frame_stack: bool = True,
                 scale: bool = False) -> None:
        self._env_id = env_id
        self._max_episode_steps = max_episode_steps
        self._episode_life = episode_life
        self._clip_rewards = clip_rewards
        self._frame_stack = frame_stack
        self._scale = scale

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = atari_wrappers.make_atari(self._env_id)
        env = GymWrapper(
            ImageObservationWrapper(
                atari_wrappers.wrap_deepmind(env, self._episode_life,
                                             self._clip_rewards,
                                             self._frame_stack, self._scale)))
        if self._max_episode_steps is not None:
            env = TimeLimitWrapper(env, self._max_episode_steps)
        return env
