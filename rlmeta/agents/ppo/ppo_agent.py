# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, RMSRescaler
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
                 batch_size: int = 128,
                 grad_clip: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 vf_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 reward_rescaling: bool = True,
                 advantage_normalization: bool = True,
                 value_clip: bool = True,
                 learning_starts: Optional[int] = None,
                 push_every_n_steps: int = 1) -> None:
        super().__init__()

        self.model = model
        self.deterministic_policy = deterministic_policy

        self.replay_buffer = replay_buffer
        self.controller = controller

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.reward_rescaling = reward_rescaling
        if self.reward_rescaling:
            self.reward_rescaler = RMSRescaler(size=1)
        self.advantage_normalization = advantage_normalization
        self.value_clip = value_clip

        self.learning_starts = learning_starts
        self.push_every_n_steps = push_every_n_steps
        self.done = False
        self.trajectory = []
        self.step_counter = 0

        self._device = None

    def reset(self) -> None:
        self.step_counter = 0

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = self.model.act(
            obs, torch.tensor([self.deterministic_policy]))
        return Action(action, info={"logpi": logpi, "v": v})

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = await self.model.async_act(
            obs, torch.tensor([self.deterministic_policy]))
        return Action(action, info={"logpi": logpi, "v": v})

    async def async_observe_init(self, timestep: TimeStep) -> None:
        obs, _, done, _ = timestep
        if done:
            self.trajectory = []
        else:
            self.trajectory = [{"obs": obs}]

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        act, info = action
        obs, reward, done, _ = next_timestep

        cur = self.trajectory[-1]
        cur["action"] = act
        cur["logpi"] = info["logpi"]
        cur["v"] = info["v"]
        cur["reward"] = reward

        if not done:
            self.trajectory.append({"obs": obs})
        self.done = done

    def update(self) -> None:
        if not self.done:
            return
        if self.replay_buffer is not None:
            replay = self._make_replay()
            self.replay_buffer.extend(replay)
        self.trajectory = []

    async def async_update(self) -> None:
        if not self.done:
            return
        if self.replay_buffer is not None:
            replay = self._make_replay()
            await self.replay_buffer.async_extend(replay)
        self.trajectory = []

    def train(self, num_steps: int) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.TRAIN, reset=True)

        self.replay_buffer.warm_up(self.learning_starts)
        stats = StatsDict()

        console.log(f"Training for num_steps = {num_steps}")
        for _ in track(range(num_steps), description="Training..."):
            t0 = time.perf_counter()
            _, batch, _ = self.replay_buffer.sample(self.batch_size)
            t1 = time.perf_counter()
            step_stats = self._train_step(batch)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.extend(step_stats)
            stats.extend(time_stats)

            self.step_counter += 1
            if self.step_counter % self.push_every_n_steps == 0:
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

    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self.model.parameters()).device
        return self._device

    def _make_replay(self) -> List[NestedTensor]:
        adv, ret = self._calculate_gae_and_return(
            [x["v"] for x in self.trajectory],
            [x["reward"] for x in self.trajectory],
            self.reward_rescaler if self.reward_rescaling else None)
        for cur, a, r in zip(self.trajectory, adv, ret):
            cur["gae"] = a
            cur["ret"] = r
            cur.pop("reward")
        return self.trajectory

    def _calculate_gae_and_return(
        self,
        values: Sequence[Union[float, torch.Tensor]],
        rewards: Sequence[Union[float, torch.Tensor]],
        reward_rescaler: Optional[Rescaler] = None
    ) -> Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
        adv = []
        ret = []
        v = torch.zeros(1)
        gae = torch.zeros(1)
        for value, reward in zip(reversed(values), reversed(rewards)):
            next_v = v
            v = value
            if reward_rescaler is not None:
                v = reward_rescaler.recover(v)
            delta = reward + self.gamma * next_v - v
            gae = delta + self.gamma * self.gae_lambda * gae
            adv.append(gae)
            ret.append(gae + v)

        if reward_rescaler is not None:
            ret = data_utils.stack_tensors(ret)
            reward_rescaler.update(ret)
            ret = reward_rescaler.rescale(ret)
            ret = ret.unbind()

        return reversed(adv), reversed(ret)

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        batch = nested_utils.map_nested(lambda x: x.to(self.device()), batch)
        self.optimizer.zero_grad()

        obs = batch["obs"]
        act = batch["action"]
        old_logpi = batch["logpi"]
        adv = batch["gae"]
        ret = batch["ret"]
        logpi, v = self._model_forward(obs)

        policy_loss, ratio = self._policy_loss(logpi.gather(dim=-1, index=act),
                                               old_logpi, adv)
        value_loss = self._value_loss(ret, v, batch.get("v", None))
        entropy = self._entropy(logpi)
        loss = policy_loss + (self.vf_loss_coeff *
                              value_loss) - (self.entropy_coeff * entropy)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.grad_clip)
        self.optimizer.step()

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
        return self.model(obs)

    def _policy_loss(self, logpi: torch.Tensor, old_logpi: torch.Tensor,
                     adv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.advantage_normalization:
            # Advantage normalization
            std, mean = torch.std_mean(adv, unbiased=False)
            adv = (adv - mean) / std

        # Policy clip
        ratio = (logpi - old_logpi).exp()
        ratio_clamp = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip)
        surr1 = ratio * adv
        surr2 = ratio_clamp * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss, ratio

    def _value_loss(self,
                    ret: torch.Tensor,
                    v: torch.Tensor,
                    old_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.value_clip:
            # Value clip
            v_clamp = old_v + (v - old_v).clamp(-self.eps_clip, self.eps_clip)
            vf1 = (ret - v).square()
            vf2 = (ret - v_clamp).square()
            value_loss = torch.max(vf1, vf2).mean() * 0.5
        else:
            value_loss = (ret - v).square().mean() * 0.5
        return value_loss

    def _entropy(self, logpi: torch.Tensor) -> torch.Tensor:
        return -(logpi.exp() * logpi).sum(dim=-1).mean()
