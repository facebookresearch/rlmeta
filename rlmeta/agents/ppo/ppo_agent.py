# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn as nn

import rlmeta_extension.nested_utils as nested_utils

from typing import Dict, List, Optional, Tuple

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_utils import StatsDict


class PPOAgent(Agent):
    def __init__(self,
                 model: ModelLike,
                 deterministic_policy: bool = False,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 controller: Optional[ControllerLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 128,
                 grad_clip: float = 50.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 entropy_ratio: float = 0.01,
                 value_clip: bool = True,
                 advantage_normalization: bool = True,
                 push_every_n_steps: int = 1) -> None:
        super(PPOAgent, self).__init__()

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
        self.entropy_ratio = entropy_ratio
        self.value_clip = value_clip
        self.advantage_normalization = advantage_normalization

        self.push_every_n_steps = push_every_n_steps
        self.done = False
        self.trajectory = []

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

    def observe_init(self, timestep: TimeStep) -> None:
        obs, _, done, _ = timestep
        if done:
            self.trajectory = []
        else:
            self.trajectory = [{"obs": obs}]

    def observe(self, action: Action, next_timestep: TimeStep) -> None:
        act, info = action
        obs, reward, done, _ = next_timestep

        cur = self.trajectory[-1]
        cur["reward"] = reward
        cur["action"] = act
        cur["logpi"] = info["logpi"]
        cur["v"] = info["v"]

        if not done:
            self.trajectory.append({"obs": obs})
        self.done = done

    def update(self) -> None:
        if not self.done:
            return
        if self.replay_buffer is not None:
            replay = self.make_replay()
            self.replay_buffer.extend(replay)
        self.trajectory = []

    async def async_update(self) -> None:
        if not self.done:
            return
        if self.replay_buffer is not None:
            replay = self.make_replay()
            await self.replay_buffer.async_extend(replay)
        self.trajectory = []

    def train(self, num_steps: int) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.TRAIN)

        self.replay_buffer.warm_up()
        stats = StatsDict()
        for step in range(num_steps):
            t0 = time.time()
            batch = self.replay_buffer.sample(self.batch_size)
            t1 = time.time()
            step_stats = self.train_step(batch)
            t2 = time.time()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.add_dict(step_stats)
            stats.add_dict(time_stats)

            if step % self.push_every_n_steps == self.push_every_n_steps - 1:
                self.model.push()

        episode_stats = self.controller.get_stats()
        self.controller.reset()
        stats.update(episode_stats)

        return stats

    def eval(self, num_episodes: Optional[int] = None) -> Optional[StatsDict]:
        self.controller.set_phase(Phase.EVAL, limit=num_episodes)
        while self.controller.get_count() < num_episodes:
            time.sleep(1)
        stats = self.controller.get_stats()
        self.controller.reset()
        return stats

    def make_replay(self) -> List[NestedTensor]:
        v = 0.0
        gae = 0.0
        for cur in reversed(self.trajectory):
            reward = cur.pop("reward")
            v_ = v
            v = cur["v"]
            delta = reward + self.gamma * v_ - v
            gae = delta + self.gamma * self.gae_lambda * gae
            cur["gae"] = gae
            cur["return"] = gae + v
        return self.trajectory

    def train_step(self, batch: NestedTensor) -> Dict[str, float]:
        device = next(self.model.parameters()).device
        batch = nested_utils.map_nested(lambda x: x.to(device), batch)
        self.optimizer.zero_grad()

        action = batch["action"]
        action_logpi = batch["logpi"]
        adv = batch["gae"]
        ret = batch["return"]
        logpi, v = self.model_forward(batch)

        if self.value_clip:
            # Value clip
            v_batch = batch["v"]
            v_clamp = v_batch + (v - v_batch).clamp(-self.eps_clip,
                                                    self.eps_clip)
            vf1 = (ret - v).square()
            vf2 = (ret - v_clamp).square()
            value_loss = torch.max(vf1, vf2).mean() * 0.5
        else:
            value_loss = (ret - v).square().mean() * 0.5

        entropy = -(logpi.exp() * logpi).sum(dim=-1).mean()
        entropy_loss = -self.entropy_ratio * entropy

        if self.advantage_normalization:
            # Advantage normalization
            std, mean = torch.std_mean(adv, unbiased=False)
            adv = (adv - mean) / std

        # Policy clip
        logpi = logpi.gather(dim=-1, index=action)
        ratio = (logpi - action_logpi).exp()
        ratio_clamp = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip)
        surr1 = ratio * adv
        surr2 = ratio_clamp * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        loss = policy_loss + value_loss + entropy_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.grad_clip)
        self.optimizer.step()

        return {
            "return": ret.detach().mean().item(),
            "entropy": entropy.detach().mean().item(),
            "policy_ratio": ratio.detach().mean().item(),
            "policy_loss": policy_loss.detach().mean().item(),
            "value_loss": value_loss.detach().mean().item(),
            "entropy_loss ": entropy_loss.detach().mean().item(),
            "loss": loss.detach().mean().item(),
            "grad_norm": grad_norm.detach().mean().item(),
        }

    def model_forward(
            self, batch: NestedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(batch["obs"])
