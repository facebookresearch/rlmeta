# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import asyncio
import copy

from concurrent.futures import Future
from typing import Any, Optional, Type, Union

import rlmeta.core.remote as remote
import rlmeta.utils.moolib_utils as moolib_utils

from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor
from rlmeta.utils.stats_dict import StatsDict


class Agent(abc.ABC):

    def reset(self) -> None:
        pass

    def act(self, timestep: TimeStep) -> Action:
        """
        Act function.
        """
        return asyncio.run(self.async_act(timestep))

    @abc.abstractmethod
    async def async_act(self, timestep: TimeStep) -> Action:
        """
        Async version of act function.
        """

    def observe_init(self, timestep: TimeStep) -> None:
        """
        Observe function for initial timestep from Env.
        """
        asyncio.run(self.async_observe_init(timestep))

    @abc.abstractmethod
    async def async_observe_init(self, timestep: TimeStep) -> None:
        """
        Async version of observe function initial timestep from Env.
        """

    def observe(self, action: Action, next_timestep: TimeStep) -> None:
        """
        Observe function for action and next timestep.
        """
        asyncio.run(self.async_observe(action, next_timestep))

    @abc.abstractmethod
    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        """
        Async version of observe function for action and next timestep.
        """

    def update(self) -> None:
        """
        Update function after each step.
        """
        asyncio.run(self.async_update())

    @abc.abstractmethod
    async def async_update(self) -> None:
        """
        Async version of update function after each step.
        """

    def connect(self) -> None:
        for obj_name in dir(self):
            obj = getattr(self, obj_name)
            if isinstance(obj, remote.Remote):
                obj.connect()

    def train(self,
              num_steps: int,
              keep_evaluation_loops: bool = False) -> StatsDict:
        pass

    def eval(self,
             num_episodes: int,
             keep_training_loops: bool = False,
             non_blocking: bool = False) -> Union[StatsDict, Future]:
        pass


class AgentFactory:

    def __init__(self, cls: Type[Agent], *args, **kwargs) -> None:
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self, index: int) -> Agent:
        args = []
        kwargs = {}
        for x in self._args:
            args.append(self._make_arg(x, index))
        for k, v in self._kwargs.items():
            kwargs[k] = self._make_arg(v, index)
        return self._cls(*args, **kwargs)

    def _make_arg(self, arg: Any, index: int) -> Any:
        if isinstance(arg, remote.Remote):
            arg = copy.deepcopy(arg)
            arg.name = moolib_utils.expend_name_by_index(arg.name, index)
        return arg
