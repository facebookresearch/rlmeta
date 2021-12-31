# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import copy

from typing import Any, Optional, Type

import rlmeta.core.remote as remote
import rlmeta.utils.moolib_utils as moolib_utils

from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor
from rlmeta.utils.stats_dict import StatsDict


class Agent(abc.ABC):

    @abc.abstractmethod
    def act(self, timestep: TimeStep) -> Action:
        """
        Act function.
        """

    async def async_act(self, timestep: TimeStep) -> Action:
        return self.act(timestep)

    @abc.abstractmethod
    def observe_init(self, timestep: TimeStep) -> None:
        """
        Observe initial timestep from Env.
        """

    async def async_observe_init(self, timestep: TimeStep) -> None:
        self.observe_init(timestep)

    @abc.abstractmethod
    def observe(self, action: Action, next_timestep: TimeStep) -> None:
        """
        Observe action and next timestep.
        """

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        self.observe(action, next_timestep)

    @abc.abstractmethod
    def update(self) -> None:
        """
        Update function after each step.
        """

    async def async_update(self) -> None:
        self.update()

    def connect(self) -> None:
        for obj_name in dir(self):
            obj = getattr(self, obj_name)
            if isinstance(obj, remote.Remote):
                obj.connect()

    def train(self, num_steps: int) -> Optional[StatsDict]:
        pass

    def eval(self, num_episodes: int) -> Optional[StatsDict]:
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
