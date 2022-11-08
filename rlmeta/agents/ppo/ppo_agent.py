# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, StdRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_dict import StatsDict

console = Console()


class PPOAgent(Agent):

    def __init__(self,
                 model: ModelLike,
                 deterministic_policy: bool = False,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 controller: Optional[ControllerLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 512,
                 max_grad_norm: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ratio_clipping_eps: float = 0.2,
                 value_clipping_eps: Optional[float] = 0.2,
                 vf_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 rescale_reward: bool = True,
                 max_abs_reward: float = 10.0,
                 normalize_advantage: bool = True,
                 learning_starts: Optional[int] = None,
                 model_push_period: int = 10,
                 local_batch_size: int = 1024) -> None:
        super().__init__()

        self._model = model
        self._deterministic_policy = torch.tensor([deterministic_policy])

        self._replay_buffer = replay_buffer
        self._controller = controller

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._ratio_clipping_eps = ratio_clipping_eps
        self._value_clipping_eps = value_clipping_eps
        self._vf_loss_coeff = vf_loss_coeff
        self._entropy_coeff = entropy_coeff
        self._rescale_reward = rescale_reward
        self._max_abs_reward = max_abs_reward
        self._reward_rescaler = StdRescaler(size=1) if rescale_reward else None
        self._normalize_advantage = normalize_advantage

        self._learning_starts = learning_starts
        self._model_push_period = model_push_period
        self._local_batch_size = local_batch_size

        self._trajectory = []
        self._step_counter = 0
        self._eval_executor = None

    def reset(self) -> None:
        self._step_counter = 0

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = self._model.act(obs, self._deterministic_policy)
        return Action(action, info={"logpi": logpi, "v": v})

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = await self._model.async_act(
            obs, self._deterministic_policy)
        return Action(action, info={"logpi": logpi, "v": v})

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

        cur = self._trajectory[-1]
        cur["action"] = act
        cur["logpi"] = info["logpi"]
        cur["v"] = info["v"]
        cur["reward"] = reward
        self._trajectory.append({"obs": obs, "done": done})

    def update(self) -> None:
        if not self._trajectory or not self._trajectory[-1]["done"]:
            return
        if self._replay_buffer is not None:
            replay = self._make_replay()
            self._send_replay(replay)
        self._trajectory.clear()

    async def async_update(self) -> None:
        if not self._trajectory or not self._trajectory[-1]["done"]:
            return
        if self._replay_buffer is not None:
            replay = await self._async_make_replay()
            await self._async_send_replay(replay)
        self._trajectory.clear()

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
            _, batch, _ = self._replay_buffer.sample(self._batch_size)
            t1 = time.perf_counter()
            step_stats = self._train_step(batch)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.extend(step_stats)
            stats.extend(time_stats)

            self._step_counter += 1
            if self._step_counter % self._model_push_period == 0:
                self._model.push()

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

    def _make_replay(self) -> List[NestedTensor]:
        self._trajectory.pop()
        adv, ret = self._calculate_gae_and_return(
            [x["v"] for x in self._trajectory],
            [x["reward"] for x in self._trajectory], self._reward_rescaler)
        for cur, a, r in zip(self._trajectory, adv, ret):
            cur["gae"] = a
            cur["ret"] = r
            cur.pop("reward")
            cur.pop("done")
        return self._trajectory

    async def _async_make_replay(self) -> List[NestedTensor]:
        return self._make_replay()

    def _send_replay(self, replay: List[NestedTensor]) -> None:
        batch = []
        while replay:
            batch.append(replay.pop())
            if len(batch) >= self._local_batch_size:
                self._replay_buffer.extend(batch)
                batch.clear()
        if batch:
            self._replay_buffer.extend(batch)
            batch.clear()

    async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
        batch = []
        while replay:
            batch.append(replay.pop())
            if len(batch) >= self._local_batch_size:
                await self._replay_buffer.async_extend(batch)
                batch.clear()
        if batch:
            await self._replay_buffer.async_extend(batch)
            batch.clear()

    def _calculate_gae_and_return(
        self,
        values: Sequence[Union[float, torch.Tensor]],
        rewards: Sequence[Union[float, torch.Tensor]],
        reward_rescaler: Optional[Rescaler] = None
    ) -> Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
        adv = []
        ret = []
        gae = torch.zeros(1)
        v = torch.zeros(1)
        g = torch.zeros(1)
        for value, reward in zip(reversed(values), reversed(rewards)):
            if isinstance(reward, float):
                reward = torch.tensor([reward])
            if reward_rescaler is not None:
                g = reward + self._gamma * g
                reward_rescaler.update(g)
                reward = reward_rescaler.rescale(reward)
                reward.clamp_(-self._max_abs_reward, self._max_abs_reward)
            delta = reward + self._gamma * v - value
            v = value
            gae = delta + self._gamma * self._gae_lambda * gae
            adv.append(gae)
            ret.append(gae + v)
        return reversed(adv), reversed(ret)

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        device = self._model.device
        batch = nested_utils.map_nested(lambda x: x.to(device), batch)
        self._optimizer.zero_grad()

        obs = batch["obs"]
        act = batch["action"]
        adv = batch["gae"]
        ret = batch["ret"]
        behavior_logpi = batch["logpi"]
        behavior_v = batch["v"]

        logpi, v = self._model_forward(obs)
        policy_loss, ratio = self._policy_loss(logpi.gather(dim=-1, index=act),
                                               behavior_logpi, adv)
        value_loss = self._value_loss(ret, v, behavior_v)
        entropy = self._entropy(logpi)
        loss = policy_loss + (self._vf_loss_coeff *
                              value_loss) - (self._entropy_coeff * entropy)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(),
                                             self._max_grad_norm)
        self._optimizer.step()

        return {
            "return": ret.detach().mean().item(),
            "policy_ratio": ratio.detach().mean().item(),
            "policy_loss": policy_loss.detach().mean().item(),
            "value_loss": value_loss.detach().mean().item(),
            "entropy": entropy.detach().mean().item(),
            "loss": loss.detach().mean().item(),
            "grad_norm": grad_norm.detach().mean().item(),
        }

    def _model_forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self._model(obs)

    def _policy_loss(self, logpi: torch.Tensor, behavior_logpi: torch.Tensor,
                     adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._normalize_advantage:
            std, mean = torch.std_mean(adv, unbiased=False)
            adv = (adv - mean) / std

        ratio = (logpi - behavior_logpi).exp()
        clipped_ratio = ratio.clamp(1.0 - self._ratio_clipping_eps,
                                    1.0 + self._ratio_clipping_eps)
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss, ratio

    def _value_loss(self,
                    ret: torch.Tensor,
                    v: torch.Tensor,
                    behavior_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._value_clipping_eps is None:
            return F.mse_loss(v, ret)

        clipped_v = behavior_v + torch.clamp(
            v - behavior_v, -self._value_clipping_eps, self._value_clipping_eps)
        vf1 = F.mse_loss(v, ret, reduction="none")
        vf2 = F.mse_loss(clipped_v, ret, reduction="none")
        return torch.max(vf1, vf2).mean()

    def _entropy(self, logpi: torch.Tensor) -> torch.Tensor:
        return -(logpi.exp() * logpi).sum(dim=-1).mean()

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
