# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

import rlmeta.utils.data_utils as data_utils
import rlmeta_extension.nested_utils as nested_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor
from rlmeta.utils.stats_dict import StatsDict


class ApeXDQNAgent(Agent):

    def __init__(self,
                 model: ModelLike,
                 eps: float = 0.1,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 controller: Optional[ControllerLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 128,
                 grad_clip: float = 50.0,
                 multi_step: int = 1,
                 gamma: float = 0.99,
                 sync_every_n_steps: int = 10,
                 push_every_n_steps: int = 1) -> None:
        super().__init__()

        self.model = model
        self.eps = eps

        self.replay_buffer = replay_buffer
        self.controller = controller

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.multi_step = multi_step
        self.gamma = gamma

        self.sync_every_n_steps = sync_every_n_steps
        self.push_every_n_steps = push_every_n_steps

        self.trajectory = []

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action = self.model.act(obs, torch.tensor([self.eps]))
        return Action(action, info=None)

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action = await self.model.async_act(obs, torch.tensor([self.eps]))
        return Action(action, info=None)

    async def async_observe_init(self, timestep: TimeStep) -> None:
        obs, _, done, _ = timestep
        self.trajectory = [{"obs": obs, "done": done}]

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        act, _ = action
        obs, reward, done, _ = next_timestep
        cur = self.trajectory[-1]
        cur["reward"] = reward
        cur["action"] = act
        self.trajectory.append({"obs": obs, "done": done})

    def update(self) -> None:
        if not self.trajectory[-1]["done"]:
            return
        if self.replay_buffer is not None:
            replay = self.make_replay()
            batch = data_utils.stack_fields(replay)
            priority = self.model.compute_priority(
                batch, torch.tensor([self.gamma**self.multi_step]))
            self.replay_buffer.extend(replay, priority)
        self.trajectory = []

    async def async_update(self) -> None:
        if not self.trajectory[-1]["done"]:
            return
        if self.replay_buffer is not None:
            replay = self.make_replay()
            batch = data_utils.stack_fields(replay)
            priority = await self.model.async_compute_priority(
                batch, torch.tensor([self.gamma**self.multi_step]))
            await self.replay_buffer.async_extend(replay, priority)
        self.trajectory = []

    def train(self, num_steps: int) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.TRAIN, reset=True)

        self.replay_buffer.warm_up()
        stats = StatsDict()
        for step in range(num_steps):
            t0 = time.perf_counter()
            batch, weight, index = self.replay_buffer.sample(self.batch_size)
            t1 = time.perf_counter()
            step_stats = self.train_step(batch, weight, index)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.add_dict(step_stats)
            stats.add_dict(time_stats)

            if step % self.sync_every_n_steps == self.sync_every_n_steps - 1:
                self.model.sync_target_net()

            if step % self.push_every_n_steps == self.push_every_n_steps - 1:
                self.model.push()

        episode_stats = self.controller.get_stats()
        stats.update(episode_stats)

        return stats

    def eval(self, num_episodes: Optional[int] = None) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.EVAL, limit=num_episodes, reset=True)
        while self.controller.get_count() < num_episodes:
            time.sleep(1)
        stats = self.controller.get_stats()
        return stats

    def make_replay(self) -> Optional[List[NestedTensor]]:
        trajectory_len = len(self.trajectory)
        if trajectory_len <= self.multi_step:
            return None

        replay = []
        append = replay.append
        for i in range(0, trajectory_len - self.multi_step):
            cur = self.trajectory[i]
            nxt = self.trajectory[i + self.multi_step]
            obs = cur["obs"]
            act = cur["action"]
            next_obs = nxt["obs"]
            done = nxt["done"]
            reward = 0.0
            for j in range(self.multi_step):
                reward += (self.gamma**j) * self.trajectory[i + j]["reward"]
            append({
                "obs": obs,
                "action": act,
                "reward": torch.tensor([reward]),
                "next_obs": next_obs,
                "done": torch.tensor([done]),
            })

        return replay

    def train_step(self, batch: NestedTensor, weight: torch.Tensor,
                   index: torch.Tensor) -> Dict[str, float]:
        device = next(self.model.parameters()).device
        # batch = nested_utils.map_nested(lambda x: x.to(device), batch)
        self.optimizer.zero_grad()

        td_err = self.model.td_error(
            batch, torch.tensor([self.gamma**self.multi_step]))
        weight = weight.to(device)  # size = (batch_size)
        loss = td_err.square() * weight * 0.5
        loss = loss.mean()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
        self.optimizer.step()
        priority = td_err.detach().abs().cpu()
        self.replay_buffer.update_priority(index, priority)

        return {
            "reward": batch["reward"].detach().mean().item(),
            "loss": loss.detach().mean().item(),
            "grad_norm": grad_norm.detach().mean().item(),
        }


class ApeXDQNAgentFactory(AgentFactory):

    def __init__(self,
                 model: ModelLike,
                 eps_func: Callable[[int], float],
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 controller: Optional[ControllerLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 128,
                 grad_clip: float = 50.0,
                 multi_step: int = 1,
                 gamma: float = 0.99,
                 sync_every_n_steps: int = 10,
                 push_every_n_steps: int = 1) -> None:
        self._model = model
        self._eps_func = eps_func
        self._replay_buffer = replay_buffer
        self._controller = controller
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._grad_clip = grad_clip
        self._multi_step = multi_step
        self._gamma = gamma
        self._sync_every_n_steps = sync_every_n_steps
        self._push_every_n_steps = push_every_n_steps

    def __call__(self, index: int):
        model = self._make_arg(self._model, index)
        eps = self._eps_func(index)
        replay_buffer = self._make_arg(self._replay_buffer, index)
        controller = self._make_arg(self._controller, index)
        return ApeXDQNAgent(model, eps, replay_buffer, controller,
                            self._optimizer, self._batch_size, self._grad_clip,
                            self._multi_step, self._gamma,
                            self._sync_every_n_steps, self._push_every_n_steps)


class ConstantEpsFunc:

    def __init__(self, eps: float) -> None:
        self._eps = eps

    def __call__(self, index: int) -> float:
        return self._eps


class FlexibleEpsFunc:

    def __init__(self, eps: float, num: int, alpha: float = 7.0) -> None:
        self._eps = eps
        self._num = num
        self._alpha = alpha

    def __call__(self, index: int) -> float:
        if self._num == 1:
            return self._eps
        return self._eps**(1.0 + index / (self._num - 1) * self._alpha)
