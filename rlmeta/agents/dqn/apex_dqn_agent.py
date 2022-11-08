# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.console import Console
from rich.progress import track

import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent, AgentFactory
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import SqrtRescaler
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
        batch_size: int = 512,
        max_grad_norm: float = 40.0,
        n_step: int = 1,
        gamma: float = 0.99,
        importance_sampling_exponent: float = 0.4,
        max_abs_reward: Optional[int] = None,
        rescale_value: bool = False,
        value_clipping_eps: Optional[float] = 0.2,
        target_sync_period: Optional[int] = None,
        learning_starts: Optional[int] = None,
        model_push_period: int = 10,
        local_batch_size: int = 1024,
        collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                      NestedTensor]] = None,
        additional_models_to_update: Optional[List[ModelLike]] = None,
    ) -> None:
        super().__init__()

        self._model = model
        self._eps = torch.tensor([eps])

        self._replay_buffer = replay_buffer
        self._controller = controller

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        self._n_step = n_step
        self._gamma = gamma
        self._gamma_pow = tuple(gamma**i for i in range(n_step + 1))
        self._importance_sampling_exponent = importance_sampling_exponent
        self._max_abs_reward = max_abs_reward
        self._value_clipping_eps = value_clipping_eps

        self._rescale_value = rescale_value
        self._rescaler = SqrtRescaler() if rescale_value else None

        self._target_sync_period = target_sync_period
        self._learning_starts = learning_starts
        self._model_push_period = model_push_period

        self._local_batch_size = local_batch_size
        self._collate_fn = torch.stack if collate_fn is None else collate_fn

        self._additional_models_to_update = additional_models_to_update

        self._step_counter = 0
        self._trajectory = []
        self._update_priorities_future = None
        self._eval_executor = None

    def reset(self) -> None:
        self._step_counter = 0

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, q, v = self._model.act(obs, self._eps)
        return Action(action, info={"q": q, "v": v})

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, q, v = await self._model.async_act(obs, self._eps)
        return Action(action, info={"q": q, "v": v})

    async def async_observe_init(self, timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return

        obs, _, done, _ = timestep
        if done:
            self._trajectory.clear()
        else:
            self._trajectory = [{"obs": obs, "done": done}]

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        act, info = action
        obs, reward, done, _ = next_timestep
        reward = torch.tensor([reward])
        if self._max_abs_reward is not None:
            reward.clamp_(-self._max_abs_reward, self._max_abs_reward)

        cur = self._trajectory[-1]
        cur["reward"] = reward
        cur["action"] = act
        cur["q"] = info["q"]
        cur["v"] = info["v"]
        self._trajectory.append({"obs": obs, "done": done})

    def update(self) -> None:
        if self._replay_buffer is None or not self._trajectory[-1]["done"]:
            return
        replay = self._make_replay()
        self._send_replay(replay)
        self._trajectory.clear()

    async def async_update(self) -> None:
        if self._replay_buffer is None or not self._trajectory[-1]["done"]:
            return
        replay = self._make_replay()
        await self._async_send_replay(replay)
        self._trajectory.clear()

    def connect(self) -> None:
        super().connect()
        if self._additional_models_to_update is not None:
            for m in self._additional_models_to_update:
                m.connect()

    def train(self,
              num_steps: int,
              keep_evaluation_loops: bool = False) -> StatsDict:
        phase = self._controller.phase()
        if keep_evaluation_loops:
            self._controller.set_phase(Phase.TRAIN | phase)
        else:
            self._controller.set_phase(Phase.TRAIN)

        self._replay_buffer.warm_up(self._learning_starts)
        stats = StatsDict()

        console.log(f"Training for num_steps = {num_steps}")
        for _ in track(range(num_steps), description="Training..."):
            t0 = time.perf_counter()
            keys, batch, probabilities = self._replay_buffer.sample(
                self._batch_size)
            t1 = time.perf_counter()
            step_stats = self._train_step(keys, batch, probabilities)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.extend(step_stats)
            stats.extend(time_stats)

            self._step_counter += 1
            if (self._target_sync_period is not None and
                    self._step_counter % self._target_sync_period == 0):
                self._model.sync_target_net()
                if self._additional_models_to_update is not None:
                    for m in self._additional_models_to_update:
                        m.sync_target_net()

            if self._step_counter % self._model_push_period == 0:
                self._model.push()
                if self._additional_models_to_update is not None:
                    for m in self._additional_models_to_update:
                        m.push()

        # Release current model to stable.
        self._model.push()
        self._model.release()

        episode_stats = self._controller.stats(Phase.TRAIN)
        stats.update(episode_stats)
        self._controller.reset_phase(Phase.TRAIN)

        return stats

    def eval(self,
             num_episodes: Optional[int] = None,
             keep_training_loops: bool = False,
             non_blocking: bool = False) -> Union[StatsDict, Future]:
        if not non_blocking:
            return self._eval(num_episodes, keep_training_loops)

        if self._eval_executor is None:
            self._eval_executor = ThreadPoolExecutor(max_workers=1)
        return self._eval_executor.submit(self._eval, num_episodes,
                                          keep_training_loops)

    def _make_replay(self) -> Optional[List[NestedTensor]]:
        trajectory_len = len(self._trajectory)
        if trajectory_len < 2:
            return None

        replay = []
        r = torch.zeros(1)
        for i in range(trajectory_len - 2, -1, -1):
            k = min(self._n_step, trajectory_len - 1 - i)
            cur = self._trajectory[i]
            nxt = self._trajectory[i + k]
            obs = cur["obs"]
            act = cur["action"]
            q = cur["q"]
            reward = cur["reward"]
            done = nxt["done"]
            v = torch.zeros(1) if done else nxt["v"]

            if self._rescaler is not None:
                v = self._rescaler.recover(v)

            gamma = torch.zeros(1) if done else torch.tensor(
                [self._gamma_pow[k]])
            r = reward + self._gamma * r - gamma * nxt.get("reward", 0.0)
            target = r + gamma * v

            if self._rescaler is not None:
                target = self._rescaler.rescale(target)

            replay.append({"obs": obs, "action": act, "q": q, "target": target})

        replay.reverse()
        return replay

    def _send_replay(self, replay: List[NestedTensor]) -> None:
        batch = []
        while replay:
            batch.append(replay.pop())
            if len(batch) >= self._local_batch_size:
                b = nested_utils.collate_nested(self._collate_fn, batch)
                priorities = self._model.compute_priority(
                    b["obs"], b["action"], b["target"])
                self._replay_buffer.extend(batch, priorities)
                batch.clear()
        if batch:
            b = nested_utils.collate_nested(self._collate_fn, batch)
            priorities = self._model.compute_priority(b["obs"], b["action"],
                                                      b["target"])
            self._replay_buffer.extend(batch, priorities)
            batch.clear()

    async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
        batch = []
        while replay:
            batch.append(replay.pop())
            if len(batch) >= self._local_batch_size:
                b = nested_utils.collate_nested(self._collate_fn, batch)
                priorities = await self._model.async_compute_priority(
                    b["obs"], b["action"], b["target"])
                await self._replay_buffer.async_extend(batch, priorities)
                batch.clear()
        if batch:
            b = nested_utils.collate_nested(self._collate_fn, batch)
            priorities = await self._model.async_compute_priority(
                b["obs"], b["action"], b["target"])
            await self._replay_buffer.async_extend(batch, priorities)
            batch.clear()

    def _train_step(self, keys: torch.Tensor, batch: NestedTensor,
                    probabilities: torch.Tensor) -> Dict[str, float]:
        device = next(self._model.parameters()).device
        batch = nested_utils.map_nested(lambda x: x.to(device), batch)
        self._optimizer.zero_grad()

        obs = batch["obs"]
        action = batch["action"]
        target = batch["target"]
        behavior_q = batch["q"]

        probabilities = probabilities.to(dtype=target.dtype, device=device)
        weight = probabilities.pow(-self._importance_sampling_exponent)
        weight.div_(weight.max())

        q = self._model.q(obs, action)
        loss = self._loss(target, q, behavior_q, weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        self._optimizer.step()

        with torch.no_grad():
            td_err = self._model.td_error(obs, action, target)
        priorities = td_err.detach().squeeze(-1).abs().cpu()

        # Wait for previous update request
        if self._update_priorities_future is not None:
            self._update_priorities_future.wait()

        # Async update to start next training step when waiting for updating
        # priorities.
        self._update_priorities_future = self._replay_buffer.async_update(
            keys, priorities)

        return {
            "td_error": td_err.detach().mean().item(),
            "loss": loss.detach().mean().item(),
            "grad_norm": grad_norm.detach().mean().item(),
        }

    def _loss(self, target: torch.Tensor, q: torch.Tensor,
              behavior_q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self._value_clipping_eps is None:
            return (F.mse_loss(q, target, reduction="none").squeeze(-1) *
                    weight).mean()

        # Apply approximate trust region value update.
        # https://arxiv.org/pdf/2209.07550.pdf
        clipped_q = behavior_q + torch.clamp(
            q - behavior_q, -self._value_clipping_eps, self._value_clipping_eps)
        err1 = F.mse_loss(q, target, reduction="none")
        err2 = F.mse_loss(clipped_q, target, reduction="none")
        return (torch.max(err1, err2).squeeze(-1) * weight).mean()

    def _eval(self,
              num_episodes: int,
              keep_training_loops: bool = False) -> StatsDict:
        phase = self._controller.phase()
        if keep_training_loops:
            self._controller.set_phase(Phase.EVAL | phase)
        else:
            self._controller.set_phase(Phase.EVAL)
        self._controller.reset_phase(Phase.EVAL, limit=num_episodes)

        while self._controller.count(Phase.EVAL) < num_episodes:
            time.sleep(1)
        stats = self._controller.stats(Phase.EVAL)

        self._controller.set_phase(phase)
        return stats


class ApexDQNAgentFactory(AgentFactory):

    def __init__(
        self,
        model: ModelLike,
        eps_func: Callable[[int], float],
        replay_buffer: Optional[ReplayBufferLike] = None,
        controller: Optional[ControllerLike] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 512,
        max_grad_norm: float = 40.0,
        n_step: int = 1,
        gamma: float = 0.99,
        importance_sampling_exponent: float = 0.4,
        max_abs_reward: Optional[int] = None,
        rescale_value: bool = False,
        value_clipping_eps: Optional[float] = 0.2,
        target_sync_period: Optional[int] = None,
        learning_starts: Optional[int] = None,
        model_push_period: int = 10,
        local_batch_size: int = 1024,
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
        self._max_grad_norm = max_grad_norm
        self._n_step = n_step
        self._gamma = gamma
        self._importance_sampling_exponent = importance_sampling_exponent
        self._max_abs_reward = max_abs_reward
        self._rescale_value = rescale_value
        self._value_clipping_eps = value_clipping_eps
        self._target_sync_period = target_sync_period
        self._learning_starts = learning_starts
        self._model_push_period = model_push_period
        self._local_batch_size = local_batch_size
        self._collate_fn = collate_fn
        self._additional_models_to_update = additional_models_to_update

    def __call__(self, index: int) -> ApexDQNAgent:
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
            self._max_grad_norm,
            self._n_step,
            self._gamma,
            self._importance_sampling_exponent,
            self._max_abs_reward,
            self._rescale_value,
            self._value_clipping_eps,
            self._target_sync_period,
            self._learning_starts,
            self._model_push_period,
            self._local_batch_size,
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
