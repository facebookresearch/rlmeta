# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np

import gym

from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.transform_observation import TransformObservation

from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.gym_wrapper import GymWrapper


def make_atari_env(
        game: str,
        mode: Optional[int] = None,
        difficulty: Optional[int] = None,
        repeat_action_probability: float = 0.25,  # sticky actions
        full_action_space: bool = False,
        max_num_frames_per_episode: Optional[int] = None,
        render_mode: Optional[str] = None,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
        clip_rewards: bool = False,
        frame_stack: Optional[int] = 4) -> Env:
    game = "ALE/" + game + "-v5"
    env = gym.make(
        game,
        mode=mode,
        difficulty=difficulty,
        obs_type="rgb",  # Convert to grayscale in AtariPreprocessing
        frameskip=1,  # NoFrameskip, max and skip in AtariPreprocessing
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_num_frames_per_episode=max_num_frames_per_episode,
        render_mode=render_mode)

    env = AtariPreprocessing(env,
                             noop_max=noop_max,
                             frame_skip=frame_skip,
                             screen_size=screen_size,
                             terminal_on_life_loss=terminal_on_life_loss,
                             grayscale_obs=grayscale_obs,
                             grayscale_newaxis=grayscale_newaxis,
                             scale_obs=scale_obs)

    if clip_rewards:
        env = TransformObservation(env, np.sign)

    if frame_stack is not None:
        env = FrameStack(env, frame_stack)

    return GymWrapper(env)


class AtariWrapperFactory(EnvFactory):

    def __init__(
            self,
            game: str,
            mode: Optional[int] = None,
            difficulty: Optional[int] = None,
            repeat_action_probability: float = 0.0,  # v4
            full_action_space: bool = False,
            max_num_frames_per_episode: Optional[int] = None,
            render_mode: Optional[str] = None,
            noop_max: int = 30,
            frame_skip: int = 4,
            screen_size: int = 84,
            terminal_on_life_loss: bool = False,
            grayscale_obs: bool = True,
            grayscale_newaxis: bool = False,
            scale_obs: bool = False,
            clip_rewards: bool = False,
            frame_stack: Optional[int] = 4) -> None:
        # AtariEnv args.
        self._game = game
        self._mode = mode
        self._difficulty = difficulty
        self._repeat_action_probability = repeat_action_probability
        self._full_action_space = full_action_space
        self._max_num_frames_per_episode = max_num_frames_per_episode
        self._render_mode = render_mode

        # AtariPreprocessing args.
        self._noop_max = noop_max
        self._frame_skip = frame_skip
        self._screen_size = screen_size
        self._terminal_on_life_loss = terminal_on_life_loss
        self._grayscale_obs = grayscale_obs
        self._grayscale_newaxis = grayscale_newaxis
        self._scale_obs = scale_obs

        # Wrappers args.
        self._clip_rewards = clip_rewards
        self._frame_stack = frame_stack

    def __call__(self, index: int, *args, **kwargs) -> Env:
        return make_atari_env(
            self._game, self._mode, self._difficulty,
            self._repeat_action_probability, self._full_action_space,
            self._max_num_frames_per_episode, self._render_mode, self._noop_max,
            self._frame_skip, self._screen_size, self._terminal_on_life_loss,
            self._grayscale_obs, self._grayscale_newaxis, self._scale_obs,
            self._clip_rewards, self._frame_stack)
