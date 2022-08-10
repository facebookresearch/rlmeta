# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor
from rlmeta.utils.stats_dict import StatsDict

console = Console()


class ApexDQNAgent(Agent):

    def __init__(
        self,
        model: ModelLike,
        eps: float = 0.1,
        replay_buffer: Optional[ReplayBufferLike] = None,
        controller: Optional[ControllerLike] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 128,
        grad_clip: float = 50.0,
        multi_step: int = 1,
        gamma: float = 0.99,
        learning_starts: Optional[int] = None,
        sync_every_n_steps: int = 10,
        push_every_n_steps: int = 1,
        collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                      NestedTensor]] = None,
        additional_models_to_update: Optional[List[ModelLike]] = None,
    ) -> None:
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
        self.learning_starts = learning_starts

        self._additional_models_to_update = additional_models_to_update
        self.sync_every_n_steps = sync_every_n_steps
        self.push_every_n_steps = push_every_n_steps

        if collate_fn is not None:
            self.collate_fn = collate_fn
        else:
            self.collate_fn = data_utils.stack_tensors

        self.trajectory = []
        self.step_counter = 0

    def reset(self) -> None:
        self.step_counter = 0

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
            batch = nested_utils.collate_nested(self.collate_fn, replay)
            priority = self.model.compute_priority(
                batch, torch.tensor([self.gamma**self.multi_step]))
            self.replay_buffer.extend(replay, priority)
        self.trajectory = []

    async def async_update(self) -> None:
        if not self.trajectory[-1]["done"]:
            return
        if self.replay_buffer is not None:
            replay = self.make_replay()
            batch = nested_utils.collate_nested(self.collate_fn, replay)
            priority = await self.model.async_compute_priority(
                batch, torch.tensor([self.gamma**self.multi_step]))
            await self.replay_buffer.async_extend(replay, priority)
        self.trajectory = []

    def connect(self) -> None:
        super().connect()
        if self._additional_models_to_update is not None:
            for m in self._additional_models_to_update:
                m.connect()

    def train(self, num_steps: int) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.TRAIN, reset=True)

        self.replay_buffer.warm_up(self.learning_starts)
        stats = StatsDict()

        console.log(f"Training for num_steps = {num_steps}")
        for _ in track(range(num_steps), description="Training..."):
            t0 = time.perf_counter()
            index, batch, weight, timestamp = self.replay_buffer.sample(
                self.batch_size)
            t1 = time.perf_counter()
            step_stats = self.train_step(index, batch, weight, timestamp)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.extend(step_stats)
            stats.extend(time_stats)

            self.step_counter += 1
            if self.step_counter % self.sync_every_n_steps == 0:
                self.model.sync_target_net()
                if self._additional_models_to_update is not None:
                    for m in self._additional_models_to_update:
                        m.sync_target_net()

            if self.step_counter % self.push_every_n_steps == 0:
                self.model.push()
                if self._additional_models_to_update is not None:
                    for m in self._additional_models_to_update:
                        m.push()

        episode_stats = StatsDict.from_dict(self.controller.get_stats())
        stats.update(episode_stats)

        return stats

    def eval(self, num_episodes: Optional[int] = None) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.EVAL, limit=num_episodes, reset=True)
        while self.controller.get_count() < num_episodes:
            time.sleep(1)
        stats = StatsDict.from_dict(self.controller.get_stats())
        return stats

    def make_replay(self) -> Optional[List[NestedTensor]]:
        trajectory_len = len(self.trajectory)
        if trajectory_len <= 2:
            return None

        replay = []
        append = replay.append
        # for i in range(0, trajectory_len - self.multi_step):
        for i in range(0, trajectory_len - 1):
            cur = self.trajectory[i]
            nxt = self.trajectory[min(i + self.multi_step, trajectory_len - 1)]
            obs = cur["obs"]
            act = cur["action"]
            next_obs = nxt["obs"]
            done = nxt["done"]
            reward = 0.0
            for j in range(min(self.multi_step, trajectory_len - 1 - i)):
                reward += (self.gamma**j) * self.trajectory[i + j]["reward"]
            append({
                "obs": obs,
                "action": act,
                "reward": torch.tensor([reward]),
                "next_obs": next_obs,
                "done": torch.tensor([done]),
            })

        return replay

    def train_step(self, index: np.ndarray, batch: NestedTensor,
                   weight: torch.Tensor,
                   timestamp: np.ndarray) -> Dict[str, float]:
        device = next(self.model.parameters()).device
        batch = nested_utils.map_nested(lambda x: x.to(device), batch)
        self.optimizer.zero_grad()

        td_err = self.model.td_error(
            batch, torch.tensor([self.gamma**self.multi_step], device=device))
        weight = weight.to(device=device,
                           dtype=td_err.dtype)  # size = (batch_size)
        loss = td_err.square() * weight * 0.5
        loss = loss.mean()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
        self.optimizer.step()
        priority = td_err.detach().abs().cpu()
        self.replay_buffer.update_priority(index, priority, timestamp)

        return {
            "reward": batch["reward"].detach().mean().item(),
            "loss": loss.detach().mean().item(),
            "grad_norm": grad_norm.detach().mean().item(),
        }


class ApexDQNAgentFactory(AgentFactory):

    def __init__(
        self,
        model: ModelLike,
        eps_func: Callable[[int], float],
        replay_buffer: Optional[ReplayBufferLike] = None,
        controller: Optional[ControllerLike] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 128,
        grad_clip: float = 50.0,
        multi_step: int = 1,
        gamma: float = 0.99,
        learning_starts: Optional[int] = None,
        sync_every_n_steps: int = 10,
        push_every_n_steps: int = 1,
        collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                      NestedTensor]] = None,
        additional_models_to_update: Optional[List[ModelLike]] = None,
    ) -> None:
        self._model = model
        self._eps_func = eps_func
        self._replay_buffer = replay_buffer
        self._controller = controller
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._grad_clip = grad_clip
        self._multi_step = multi_step
        self._gamma = gamma
        self._learning_starts = learning_starts
        self._sync_every_n_steps = sync_every_n_steps
        self._push_every_n_steps = push_every_n_steps
        self._collate_fn = collate_fn
        self._additional_models_to_update = additional_models_to_update

    def __call__(self, index: int):
        model = self._make_arg(self._model, index)
        eps = self._eps_func(index)
        replay_buffer = self._make_arg(self._replay_buffer, index)
        controller = self._make_arg(self._controller, index)
        return ApexDQNAgent(
            model,
            eps,
            replay_buffer,
            controller,
            self._optimizer,
            self._batch_size,
            self._grad_clip,
            self._multi_step,
            self._gamma,
            self._learning_starts,
            self._sync_every_n_steps,
            self._push_every_n_steps,
            self._collate_fn,
            additional_models_to_update=self._additional_models_to_update)


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
