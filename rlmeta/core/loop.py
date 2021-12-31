# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
import logging
import time

from typing import Dict, List, NoReturn, Optional, Sequence, Union

import torch
import torch.multiprocessing as mp

import moolib

import rlmeta.core.remote as remote
import rlmeta.utils.asycio_utils as asycio_utils
import rlmeta.utils.moolib_utils as moolib_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.launchable import Launchable
from rlmeta.envs.env import Env, EnvFactory


class Loop(abc.ABC):

    @abc.abstractmethod
    def run(self, num_episodes: Optional[int] = None) -> None:
        """
        """


class AsyncLoop(Loop, Launchable):

    def __init__(self,
                 env_factory: EnvFactory,
                 agent_factory: AgentFactory,
                 controller: ControllerLike,
                 running_phase: Phase,
                 should_update: bool = False,
                 num_rollouts: int = 1,
                 index: int = 0,
                 index_offset: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self._running_phase = running_phase
        self._should_update = should_update
        self._index = index
        self._num_rollouts = num_rollouts
        if index_offset is None:
            self._index_offset = index * num_rollouts
        else:
            self._index_offset = index_offset
        self._seed = seed

        self._env_factory = env_factory
        self._agent_factory = agent_factory
        self._envs = []
        self._agents = []
        self._controller = controller

        self._loop = None
        self._tasks = []
        self._running = False

    @property
    def running_phase(self) -> Phase:
        return self._running_phase

    @property
    def should_update(self) -> bool:
        return self._should_update

    @property
    def num_rollouts(self) -> int:
        return self._num_rollouts

    @property
    def index(self) -> int:
        return self._index

    @property
    def index_offset(self) -> int:
        return self._index_offset

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def running(self) -> bool:
        return self._running

    @running.setter
    def running(self, running: bool) -> None:
        self._running = running

    def init_launching(self) -> None:
        pass

    def init_execution(self) -> None:
        for obj_name in dir(self):
            obj = getattr(self, obj_name)
            if isinstance(obj, remote.Remote):
                obj.name = moolib_utils.expend_name_by_index(
                    obj.name, self.index)
                obj.connect()
        for obj_name in dir(self):
            obj = getattr(self, obj_name)
            if isinstance(obj, Launchable):
                obj.init_execution()

        for i in range(self._num_rollouts):
            env = self._env_factory(self.index_offset + i)
            if self.seed is not None:
                env.seed(self.seed + self.index_offset + i)
            self._envs.append(env)

        for i in range(self._num_rollouts):
            agent = self._agent_factory(self.index_offset + i)
            agent.connect()
            # if self.seed is not None:
            #     agent.seed(self.seed + self.index_offset + i)
            self._agents.append(agent)

    def run(self) -> NoReturn:
        self._loop = asyncio.get_event_loop()
        self._tasks.append(
            asycio_utils.create_task(self._loop, self._check_phase()))
        for i, (env, agent) in enumerate(zip(self._envs, self._agents)):
            task = asycio_utils.create_task(
                self._loop, self._run_loop(env, agent, self.index_offset + i))
            self._tasks.append(task)
        try:
            self._loop.run_forever()
        except Exception as e:
            logging.error(e)
            raise e
        finally:
            for task in self._tasks:
                task.cancel()
            self._loop.stop()

    async def _check_phase(self) -> NoReturn:
        while True:
            cur_phase = await self._controller.async_get_phase()
            self._running = (cur_phase == self.running_phase)
            await asyncio.sleep(1)

    async def _run_loop(self,
                        env: Env,
                        agent: Agent,
                        index: int = 0) -> NoReturn:
        while True:
            while not self.running:
                await asyncio.sleep(1)
            stats = await self._run_episode(env, agent, index)
            if stats is not None:
                await self._controller.async_add_episode(stats)

    # Similar loop as DeepMind's Acme
    # https://github.com/deepmind/acme/blob/master/acme/environment_loop.py#L68
    async def _run_episode(self,
                           env: Env,
                           agent: Agent,
                           index: int = 0) -> Optional[Dict[str, float]]:
        episode_length = 0
        episode_return = 0.0

        start_time = time.perf_counter()

        timestep = env.reset()
        await agent.async_observe_init(timestep)

        while not timestep.done:
            if not self.running:
                return None
            action = await agent.async_act(timestep)
            timestep = env.step(action)
            await agent.async_observe(action, timestep)
            if self.should_update:
                await agent.async_update()

            episode_length += 1
            episode_return += timestep.reward

        episode_time = time.perf_counter() - start_time
        steps_per_second = episode_length / episode_time

        return {
            "episode_length": float(episode_length),
            "episode_return": episode_return,
            "episode_time/s": episode_time,
            "steps_per_second": steps_per_second,
        }


class ParallelLoop(Loop):

    def __init__(self,
                 env_factory: EnvFactory,
                 agent_factory: AgentFactory,
                 controller: Union[Controller, remote.Remote],
                 running_phase: Phase,
                 should_update: bool = False,
                 num_rollouts: int = 1,
                 num_workers: Optional[int] = None,
                 index: int = 0,
                 index_offset: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self._running_phase = running_phase
        self._should_update = should_update
        self._index = index
        self._num_rollouts = num_rollouts
        self._num_workers = min(mp.cpu_count(), self._num_rollouts)
        if num_workers is not None:
            self._num_workers = min(self._num_workers, num_workers)
        if index_offset is None:
            self._index_offset = index * num_rollouts
        else:
            self._index_offset = index_offset
        self._seed = seed

        self._env_factory = env_factory
        self._agent_factory = agent_factory
        self._controller = controller

        self._workloads = self._compute_workloads()
        self._async_loops = []
        self._processes = []

        index_offset = self._index_offset
        for i, workload in enumerate(self._workloads):
            loop = AsyncLoop(self._env_factory, self._agent_factory,
                             self._controller, self.running_phase,
                             self.should_update, workload, i, index_offset,
                             self.seed)
            self._async_loops.append(loop)
            index_offset += workload

    @property
    def running_phase(self) -> Phase:
        return self._running_phase

    @property
    def should_update(self) -> bool:
        return self._should_update

    @property
    def num_rollouts(self) -> int:
        return self._num_rollouts

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def index(self) -> int:
        return self._index

    @property
    def index_offset(self) -> int:
        return self._index_offset

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def run(self) -> NoReturn:
        processes = []
        for loop in self._async_loops:
            loop.init_launching()
            process = mp.Process(target=self._run_async_loop, args=(loop,))
            processes.append(process)
        for process in processes:
            process.start()
        self._processes = processes

    def start(self) -> None:
        self.run()

    def join(self) -> None:
        for process in self._processes:
            process.join()

    def terminate(self) -> None:
        for process in self._processes:
            process.terminate()

    def _compute_workloads(self) -> List[int]:
        workload = self.num_rollouts // self.num_workers
        r = self.num_rollouts % self.num_workers
        workloads = [workload + 1] * r + [workload] * (self.num_workers - r)
        return workloads

    def _run_async_loop(self, loop: AsyncLoop) -> NoReturn:
        if loop.seed is not None:
            torch.manual_seed(loop.seed + loop.index_offset)
        loop.init_execution()
        loop.run()


class LoopList:

    def __init__(self, loops: Optional[Sequence[Loop]] = None) -> None:
        self._loops = []
        if loops is not None:
            self._loops.extend(loops)

    @property
    def loops(self) -> List[Loop]:
        return self._loops

    def append(self, loop: Loop) -> None:
        self.loops.append(loop)

    def extend(self, loops: Union[LoopList, Sequence[Loop]]) -> None:
        if isinstance(loops, LoopList):
            self.loops.extend(loops.loops)
        else:
            self.loops.extend(loops)

    def start(self) -> None:
        for loop in self.loops:
            loop.start()

    def join(self) -> None:
        for loop in self.loops:
            loop.join()

    def terminate(self) -> None:
        for loop in self.loops:
            loop.terminate()


LoopLike = Union[Loop, LoopList]
