import asyncio
import time

from typing import Optional

import numpy as np

import torch
import torch.multiprocessing as mp

import rlmeta.utils.remote_utils as remote_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, Phase
from rlmeta.core.loop import ParallelLoop
from rlmeta.core.server import Server
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env, EnvFactory


class MockEnv(Env):

    def __init__(self,
                 index: int,
                 observation_space: int = 4,
                 action_space: int = 4,
                 episode_length: int = 10) -> None:
        self.index = index
        self.observation_space = observation_space
        self.action_space = action_space
        self.episode_length = episode_length
        self.step_counter = 0

    def reset(self, *args, **kwargs) -> TimeStep:
        print(f"[Env {self.index}] reset")
        print("")
        self.step_counter = 0
        obs = torch.randn(self.observation_space)
        info = {"step_counter": 0}
        return TimeStep(obs, done=False, info=info)

    def step(self, action: Action) -> TimeStep:
        self.step_counter += 1
        time.sleep(1.0)
        obs = torch.randn(self.observation_space)
        reward = np.random.randn()
        done = self.step_counter == self.episode_length
        info = {"step_counter": self.step_counter}
        print(
            f"[Env {self.index}] step = {self.step_counter}, reward = {reward}")
        print("")
        return TimeStep(obs, reward, done, info)

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        pass


class MockAgent(Agent):

    def __init__(self, index: int, action_space: int = 4) -> None:
        self.index = index
        self.action_space = action_space

    async def async_act(self, timestep: TimeStep) -> Action:
        _, reward, _, info = timestep
        step_counter = info["step_counter"]
        await asyncio.sleep(1.0)
        act = np.random.randint(self.action_space)
        print(f"[Agent {self.index}] step = {step_counter}, action = {act}")
        return Action(act)

    async def async_observe_init(self, timestep: TimeStep) -> None:
        pass

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        pass

    async def async_update(self) -> None:
        pass


def env_factory(index: int) -> MockEnv:
    return MockEnv(index)


def agent_factory(index: int) -> MockAgent:
    return MockAgent(index)


def main() -> None:
    server = Server("server", "127.0.0.1:4411")
    ctrl = Controller()
    server.add_service(ctrl)
    loop_ctrl = remote_utils.make_remote(ctrl, server)
    main_ctrl = remote_utils.make_remote(ctrl, server)
    loop = ParallelLoop(env_factory,
                        agent_factory,
                        loop_ctrl,
                        running_phase=Phase.EVAL,
                        num_rollouts=2,
                        num_workers=1)

    server.start()
    loop.start()
    main_ctrl.connect()
    main_ctrl.set_phase(Phase.EVAL, reset=True)
    time.sleep(30)
    loop.terminate()
    server.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
