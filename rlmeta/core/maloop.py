# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
import copy
import logging
import time

from typing import Dict, List, NoReturn, Optional, Sequence, Union
from rich.console import Console

import torch
import torch.multiprocessing as mp

import moolib

import rlmeta.core.remote as remote
import rlmeta.utils.asycio_utils as asycio_utils
import rlmeta.utils.moolib_utils as moolib_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.launchable import Launchable
from rlmeta.envs.env import Env, EnvFactory

console = Console()


class MALoop(abc.ABC):

    @abc.abstractmethod
    def run(self, num_episodes: Optional[int] = None) -> None:
        """
        """


class MAAsyncLoop(MALoop, Launchable):

    def __init__(self,
                 env_factory: EnvFactory,
                 agent_factory, #dictionary of agent Factory/Agent: Dict(AgentFactory)
                 controller: ControllerLike,
                 running_phase: Phase,
                 should_update: bool = False,
                 num_rollouts: int = 1,
                 index: int = 0,
                 index_offset: Optional[int] = None,
                 seed: Optional[int] = None,
                 episode_callbacks: Optional[EpisodeCallbacks] = None) -> None:
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

        self._episode_callbacks = episode_callbacks

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
        for i in range(self._num_rollouts):
            env = self._env_factory(self.index_offset + i)
            if self.seed is not None:
                env.seed(self.seed + self.index_offset + i)
            self._envs.append(env)

        for i in range(self._num_rollouts):
            agents = {}
            for k,v in self._agent_factory.items():
                agents[k] = v(self.index_offset + i)
                agents[k].connect()
            self._agents.append(agents)

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

    def run(self) -> NoReturn:
        console.log(f"Starting async loop with: {self._controller}")
        self._loop = asyncio.get_event_loop()
        self._tasks.append(
            asycio_utils.create_task(self._loop, self._check_phase()))
        for i, (env, agent) in enumerate(zip(self._envs, self._agents)):
            index = self.index_offset + i
            task = asycio_utils.create_task(
                self._loop,
                self._run_loop(index, env, agent,
                               copy.deepcopy(self._episode_callbacks)))
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
            self._running = (cur_phase == self._running_phase)
            await asyncio.sleep(1)

    async def _run_loop(
            self,
            index: int,
            env: Env,
            agent: Agent, #TODO type check
            episode_callbacks: Optional[EpisodeCallbacks] = None) -> NoReturn:
        while True:
            while not self.running:
                await asyncio.sleep(1)
            stats = await self._run_episode(index, env, agent,
                                            episode_callbacks)
            if stats is not None:
                await self._controller.async_add_episode(
                    self._running_phase, stats)

    # The method _run_episode is adapted from Acme's Enviroment.run_episode:
    # https://github.com/deepmind/acme/blob/df961057bcd2e1436d5f894ebced62d694225034/acme/environment_loop.py#L76
    #
    # It was released under the Apache License, Version 2.0 (the "License"),
    # available at:
    # http://www.apache.org/licenses/LICENSE-2.0
    async def _run_episode(
        self,
        index: int,
        env: Env,
        agent: Agent, #TODO:type check
        episode_callbacks: Optional[EpisodeCallbacks] = None
    ) -> Optional[Dict[str, float]]:
        
        episode_length = 0.0
        episode_return = {}
        for k,v in agent.items():
            episode_return[k] = 0.0
        start_time = time.perf_counter()
        if episode_callbacks is not None:
            episode_callbacks.reset()
            episode_callbacks.on_episode_start(index)

        timestep = env.reset()
        for k,v in agent.items():
            if env.action_mask[k]:
                if timestep[k].info is None:
                    timestep[k].info = {}
                timestep[k].info.update({"episode_reset":True})
                await agent[k].async_observe_init(timestep[k])
        if episode_callbacks is not None:
            episode_callbacks.on_episode_init(index, timestep)

        while not timestep["__all__"].done:
            if not self.running:
                return None
            action = {}
            for k,v in agent.items():
                action[k] = await v.async_act(timestep[k])
            timestep = env.step(action)
            
            for k,v in agent.items():
                if env.action_mask[k]:
                    await v.async_observe(action[k], timestep[k])
            if self.should_update:
                for k,v in agent.items():
                    if env.action_mask[k]:
                        await agent[k].async_update()

            episode_length += 1
            for k, v in agent.items():
                episode_return[k] += timestep[k].reward
            if episode_callbacks is not None:
                episode_callbacks.on_episode_step(index, episode_length - 1,
                                                  action, timestep)

        episode_time = time.perf_counter() - start_time
        steps_per_second = episode_length / episode_time
        if episode_callbacks is not None:
            episode_callbacks.on_episode_end(index)

        metrics = {
            "episode_length": float(episode_length),
            "episode_time/s": episode_time,
            "steps_per_second": steps_per_second,
        }
        for k,v in agent.items():
            metrics.update({k+"_episode_return": episode_return[k]})
        if episode_callbacks is not None:
            metrics.update(episode_callbacks.custom_metrics)

        return metrics


class MAParallelLoop(MALoop):

    def __init__(self,
                 env_factory: EnvFactory,
                 agent_factory: AgentFactory,#TODO type check
                 controller: Union[Controller, remote.Remote],
                 running_phase: Phase,
                 should_update: bool = False,
                 num_rollouts: int = 1,
                 num_workers: Optional[int] = None,
                 index: int = 0,
                 index_offset: Optional[int] = None,
                 seed: Optional[int] = None,
                 episode_callbacks: Optional[EpisodeCallbacks] = None) -> None:
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

        self._env_factory = env_factory
        self._agent_factory = agent_factory
        self._controller = controller
        self._seed = seed
        self._episode_callbacks = episode_callbacks

        self._workloads = self._compute_workloads()
        self._async_loops = []
        self._processes = []

        index_offset = self._index_offset
        for i, workload in enumerate(self._workloads):
            loop = MAAsyncLoop(self._env_factory, self._agent_factory,
                             self._controller, self._running_phase,
                             self._should_update, workload, i, index_offset,
                             self._seed, self._episode_callbacks)
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
        self.start()
        self.join()

    def start(self) -> None:
        processes = []
        for loop in self._async_loops:
            loop.init_launching()
            process = mp.Process(target=self._run_async_loop, args=(loop,))
            processes.append(process)
        for process in processes:
            process.start()
        self._processes = processes

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

    def __init__(self, loops: Optional[Sequence[MALoop]] = None) -> None:
        self._loops = []
        if loops is not None:
            self._loops.extend(loops)

    @property
    def loops(self) -> List[Loop]:
        return self._loops

    def append(self, loop: MALoop) -> None:
        self.loops.append(loop)

    def extend(self, loops: Union[LoopList, Sequence[MALoop]]) -> None:
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


MALoopLike = Union[MALoop, LoopList]
