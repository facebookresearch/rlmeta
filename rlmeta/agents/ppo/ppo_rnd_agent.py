# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.controller import ControllerLike
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import StdRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor


class PPORNDAgent(PPOAgent):

    def __init__(
        self,
        model: ModelLike,
        deterministic_policy: bool = False,
        replay_buffer: Optional[ReplayBufferLike] = None,
        controller: Optional[ControllerLike] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 128,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ratio_clipping_eps: float = 0.2,
        value_clipping_eps: Optional[float] = 0.2,
        intrinsic_advantage_coeff: float = 0.5,
        vf_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        rescale_reward: bool = True,
        max_abs_reward: float = 10.0,
        normalize_advantage: bool = True,
        learning_starts: Optional[int] = None,
        model_push_period: int = 10,
        local_batch_size: int = 1024,
        collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                      NestedTensor]] = None
    ) -> None:
        super().__init__(model, deterministic_policy, replay_buffer, controller,
                         optimizer, batch_size, max_grad_norm, gamma,
                         gae_lambda, ratio_clipping_eps, value_clipping_eps,
                         vf_loss_coeff, entropy_coeff, rescale_reward,
                         max_abs_reward, normalize_advantage, learning_starts,
                         model_push_period, local_batch_size)

        self._intrinsic_advantage_coeff = intrinsic_advantage_coeff

        self._reward_rescaler = None
        self._ext_reward_rescaler = StdRescaler(
            size=1) if rescale_reward else None
        self._int_reward_rescaler = StdRescaler(
            size=1) if rescale_reward else None

        self._collate_fn = torch.stack if collate_fn is None else collate_fn

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, ext_v, int_v = self._model.act(
            obs, self._deterministic_policy)
        return Action(action,
                      info={
                          "logpi": logpi,
                          "ext_v": ext_v,
                          "int_v": int_v,
                      })

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, ext_v, int_v = await self._model.async_act(
            obs, self._deterministic_policy)
        return Action(action,
                      info={
                          "logpi": logpi,
                          "ext_v": ext_v,
                          "int_v": int_v,
                      })

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return

        act, info = action
        obs, reward, done, _ = next_timestep

        cur = self._trajectory[-1]
        cur["reward"] = reward
        cur["action"] = act
        cur["logpi"] = info["logpi"]
        cur["ext_v"] = info["ext_v"]
        cur["int_v"] = info["int_v"]
        self._trajectory.append({"obs": obs, "done": done})

    def _make_replay(self) -> List[NestedTensor]:
        next_obs = [
            self._trajectory[i]["obs"] for i in range(1, len(self._trajectory))
        ]
        int_rewards = self._compute_intrinsic_rewards(next_obs,
                                                      done_at_last=True)
        return self._make_replay_impl(int_rewards)

    async def _async_make_replay(self) -> List[NestedTensor]:
        next_obs = [
            self._trajectory[i]["obs"] for i in range(1, len(self._trajectory))
        ]
        int_rewards = await self._async_compute_intrinsic_rewards(
            next_obs, done_at_last=True)
        return self._make_replay_impl(int_rewards)

    def _make_replay_impl(
            self,
            intrinsic_rewards: Sequence[NestedTensor]) -> List[NestedTensor]:
        self._trajectory.pop()

        ext_adv, ext_ret = self._calculate_gae_and_return(
            [x["ext_v"] for x in self._trajectory],
            [x["reward"] for x in self._trajectory], self._ext_reward_rescaler)
        int_adv, int_ret = self._calculate_gae_and_return(
            [x["int_v"] for x in self._trajectory], intrinsic_rewards,
            self._int_reward_rescaler)

        for cur, ext_a, ext_r, int_a, int_r in zip(self._trajectory, ext_adv,
                                                   ext_ret, int_adv, int_ret):
            cur["ext_gae"] = ext_a
            cur["ext_ret"] = ext_r
            cur["int_gae"] = int_a
            cur["int_ret"] = int_r
            cur.pop("reward")
            cur.pop("done")

        return self._trajectory

    def _compute_intrinsic_rewards(
            self,
            obs: Sequence[NestedTensor],
            done_at_last: bool = True) -> List[torch.Tensor]:
        int_rewards = []
        n = len(obs)
        obs = nested_utils.collate_nested(self._collate_fn, obs)
        for i in range(0, n, self._local_batch_size):
            batch = nested_utils.map_nested(
                lambda x, i=i: x[i:i + self._local_batch_size], obs)
            cur_rewards = self._model.intrinsic_reward(batch)
            int_rewards.extend(torch.unbind(cur_rewards))
        if done_at_last:
            int_rewards[-1].zero_()
        return int_rewards

    async def _async_compute_intrinsic_rewards(
            self,
            obs: Sequence[NestedTensor],
            done_at_last: bool = True) -> List[torch.Tensor]:
        int_rewards = []
        n = len(obs)
        obs = nested_utils.collate_nested(self._collate_fn, obs)
        for i in range(0, n, self._local_batch_size):
            batch = nested_utils.map_nested(
                lambda x, i=i: x[i:i + self._local_batch_size], obs)
            cur_rewards = await self._model.async_intrinsic_reward(batch)
            int_rewards.extend(torch.unbind(cur_rewards))
        if done_at_last:
            int_rewards[-1].zero_()
        return int_rewards

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        batch = nested_utils.map_nested(lambda x: x.to(self._model.device),
                                        batch)
        self._optimizer.zero_grad()

        obs = batch["obs"]
        act = batch["action"]
        ext_adv = batch["ext_gae"]
        ext_ret = batch["ext_ret"]
        int_adv = batch["int_gae"]
        int_ret = batch["int_ret"]
        behavior_logpi = batch["logpi"]
        behavior_ext_v = batch["ext_v"]
        behavior_int_v = batch["int_v"]

        logpi, ext_v, int_v = self._model_forward(obs)
        adv = ext_adv + self._intrinsic_advantage_coeff * int_adv
        policy_loss, ratio = self._policy_loss(logpi.gather(dim=-1, index=act),
                                               behavior_logpi, adv)

        ext_value_loss = self._value_loss(ext_ret, ext_v, behavior_ext_v)
        int_value_loss = self._value_loss(int_ret, int_v, behavior_int_v)
        value_loss = ext_value_loss + int_value_loss
        entropy = self._entropy(logpi)
        rnd_loss = self._rnd_loss(obs)

        loss = policy_loss + (self._vf_loss_coeff * value_loss) - (
            self._entropy_coeff * entropy) + rnd_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(),
                                             self._max_grad_norm)
        self._optimizer.step()

        return {
            "ext_return": ext_ret.detach().mean().item(),
            "int_return": int_ret.detach().mean().item(),
            "policy_ratio": ratio.detach().mean().item(),
            "policy_loss": policy_loss.detach().mean().item(),
            "ext_value_loss": ext_value_loss.detach().mean().item(),
            "int_value_loss": int_value_loss.detach().mean().item(),
            "value_loss": value_loss.detach().mean().item(),
            "entropy": entropy.detach().mean().item(),
            "rnd_loss": rnd_loss.detach().mean().item(),
            "loss": loss.detach().mean().item(),
            "grad_norm": grad_norm.detach().mean().item(),
        }

    def _rnd_loss(self, next_obs: torch.Tensor) -> torch.Tensor:
        return self._model.rnd_loss(next_obs)
