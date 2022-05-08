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
from rlmeta.core.rescalers import Rescaler, MomentsRescaler, RMSRescaler
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
        grad_clip: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        intrinsic_advantage_coeff: float = 0.5,
        vf_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        advantage_normalization: bool = True,
        reward_rescaling: bool = True,
        value_clip: bool = True,
        learning_starts: Optional[int] = None,
        push_every_n_steps: int = 1,
        collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                      NestedTensor]] = None
    ) -> None:
        super().__init__(model, deterministic_policy, replay_buffer, controller,
                         optimizer, batch_size, grad_clip, gamma, gae_lambda,
                         eps_clip, vf_loss_coeff, entropy_coeff,
                         advantage_normalization, reward_rescaling, value_clip,
                         learning_starts, push_every_n_steps)

        self.intrinsic_advantage_coeff = intrinsic_advantage_coeff

        if self.reward_rescaling:
            self.reward_rescaler = None
            self.ext_reward_rescaler = RMSRescaler(size=1)
            self.int_reward_rescaler = RMSRescaler(size=1)

        if collate_fn is not None:
            self.collate_fn = collate_fn
        else:
            self.collate_fn = data_utils.stack_tensors

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, ext_v, int_v = self.model.act(
            obs, torch.tensor([self.deterministic_policy]))
        return Action(action,
                      info={
                          "logpi": logpi,
                          "ext_v": ext_v,
                          "int_v": int_v,
                      })

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, ext_v, int_v = await self.model.async_act(
            obs, torch.tensor([self.deterministic_policy]))
        return Action(action,
                      info={
                          "logpi": logpi,
                          "ext_v": ext_v,
                          "int_v": int_v,
                      })

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        act, info = action
        obs, reward, done, _ = next_timestep

        cur = self.trajectory[-1]
        cur["reward"] = reward
        cur["action"] = act
        cur["logpi"] = info["logpi"]
        cur["ext_v"] = info["ext_v"]
        cur["int_v"] = info["int_v"]
        cur["next_obs"] = obs

        if not done:
            self.trajectory.append({"obs": obs})
        self.done = done

    def _make_replay(self) -> List[NestedTensor]:
        next_obs = [x["next_obs"] for x in self.trajectory]
        next_obs = self.collate_fn(next_obs)
        int_rewards = self.model.intrinsic_reward(next_obs)

        ext_adv, ext_ret = self._calculate_gae_and_return(
            [x["ext_v"] for x in self.trajectory],
            [x["reward"] for x in self.trajectory],
            self.ext_reward_rescaler if self.reward_rescaling else None)
        int_adv, int_ret = self._calculate_gae_and_return(
            [x["int_v"] for x in self.trajectory], torch.unbind(int_rewards),
            self.int_reward_rescaler if self.reward_rescaling else None)

        for cur, ext_a, ext_r, int_a, int_r in zip(self.trajectory, ext_adv,
                                                   ext_ret, int_adv, int_ret):
            cur["ext_gae"] = ext_a
            cur["ext_ret"] = ext_r
            cur["int_gae"] = int_a
            cur["int_ret"] = int_r
            cur.pop("reward")

        return self.trajectory

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        batch = nested_utils.map_nested(lambda x: x.to(self.device()), batch)
        self.optimizer.zero_grad()

        obs = batch["obs"]
        act = batch["action"]
        old_logpi = batch["logpi"]
        ext_adv = batch["ext_gae"]
        ext_ret = batch["ext_ret"]
        int_adv = batch["int_gae"]
        int_ret = batch["int_ret"]
        next_obs = batch["next_obs"]
        logpi, ext_v, int_v = self._model_forward(obs)

        adv = ext_adv + self.intrinsic_advantage_coeff * int_adv
        policy_loss, ratio = self._policy_loss(logpi.gather(dim=-1, index=act),
                                               old_logpi, adv)

        ext_value_loss = self._value_loss(ext_ret, ext_v,
                                          batch.get("ext_v", None))
        int_value_loss = self._value_loss(int_ret, int_v,
                                          batch.get("int_v", None))
        value_loss = ext_value_loss + int_value_loss
        entropy = self._entropy(logpi)
        rnd_loss = self._rnd_loss(next_obs)

        loss = policy_loss + (self.vf_loss_coeff * value_loss) - (
            self.entropy_coeff * entropy) + rnd_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.grad_clip)
        self.optimizer.step()

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
        return self.model.rnd_loss(next_obs)
