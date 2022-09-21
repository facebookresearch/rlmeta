# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import time

import hydra

import torch
import torch.multiprocessing as mp

import rlmeta.envs.atari_wrappers as atari_wrappers
import rlmeta.envs.gym_wrappers as gym_wrappers
import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.remote_utils as remote_utils

from examples.atari.dqn.atari_dqn_model import AtariDQNModel
from rlmeta.agents.dqn import (ApexDQNAgent, ApexDQNAgentFactory,
                               ConstantEpsFunc, FlexibleEpsFunc)
from rlmeta.core.controller import Phase, Controller
from rlmeta.core.loop import LoopList, ParallelLoop
from rlmeta.core.model import wrap_downstream_model
from rlmeta.core.replay_buffer import ReplayBuffer, make_remote_replay_buffer
from rlmeta.core.server import Server, ServerList
from rlmeta.samplers import PrioritizedSampler
from rlmeta.storage import TensorCircularBuffer
from rlmeta.utils.optimizer_utils import make_optimizer


@hydra.main(config_path="./conf", config_name="conf_apex_dqn")
def main(cfg):
    logging.info(hydra_utils.config_to_json(cfg))

    env = atari_wrappers.make_atari(cfg.env)
    train_model = AtariDQNModel(env.action_space.n).to(cfg.train_device)
    optimizer = make_optimizer(cfg.optimizer.name, train_model.parameters(),
                               cfg.optimizer.args)

    infer_model = copy.deepcopy(train_model).to(cfg.infer_device)

    ctrl = Controller()
    rb = ReplayBuffer(
        TensorCircularBuffer(cfg.replay_buffer_size),
        PrioritizedSampler(priority_exponent=cfg.priority_exponent))

    m_server = Server(cfg.m_server_name, cfg.m_server_addr)
    r_server = Server(cfg.r_server_name, cfg.r_server_addr)
    c_server = Server(cfg.c_server_name, cfg.c_server_addr)
    m_server.add_service(infer_model)
    r_server.add_service(rb)
    c_server.add_service(ctrl)
    servers = ServerList([m_server, r_server, c_server])

    a_model = wrap_downstream_model(train_model, m_server)
    t_model = remote_utils.make_remote(infer_model, m_server)
    e_model = remote_utils.make_remote(infer_model, m_server)

    a_ctrl = remote_utils.make_remote(ctrl, c_server)
    t_ctrl = remote_utils.make_remote(ctrl, c_server)
    e_ctrl = remote_utils.make_remote(ctrl, c_server)

    a_rb = make_remote_replay_buffer(rb,
                                     r_server,
                                     prefetch=cfg.prefetch,
                                     timeout=120)
    t_rb = make_remote_replay_buffer(rb, r_server)

    t_env_fac = gym_wrappers.AtariWrapperFactory(
        cfg.env, max_episode_steps=cfg.max_episode_steps, clip_rewards=True)
    e_env_fac = gym_wrappers.AtariWrapperFactory(
        cfg.env, max_episode_steps=cfg.max_episode_steps, clip_rewards=False)

    agent = ApexDQNAgent(
        a_model,
        replay_buffer=a_rb,
        controller=a_ctrl,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        multi_step=cfg.multi_step,
        importance_sampling_exponent=cfg.importance_sampling_exponent,
        learning_starts=cfg.get("learning_starts", None),
        sync_every_n_steps=cfg.sync_every_n_steps,
        push_every_n_steps=cfg.push_every_n_steps)
    t_agent_fac = ApexDQNAgentFactory(t_model,
                                      FlexibleEpsFunc(cfg.train_eps,
                                                      cfg.num_train_rollouts),
                                      replay_buffer=t_rb,
                                      multi_step=cfg.multi_step,
                                      reward_rescaling=False)
    e_agent_fac = ApexDQNAgentFactory(e_model, ConstantEpsFunc(cfg.eval_eps))

    t_loop = ParallelLoop(t_env_fac,
                          t_agent_fac,
                          t_ctrl,
                          running_phase=Phase.TRAIN,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.train_seed)
    e_loop = ParallelLoop(e_env_fac,
                          e_agent_fac,
                          e_ctrl,
                          running_phase=Phase.EVAL,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=cfg.eval_seed)
    loops = LoopList([t_loop, e_loop])

    servers.start()
    loops.start()
    agent.connect()

    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        stats = agent.train(cfg.steps_per_epoch)
        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))
        time.sleep(1)

        stats = agent.eval(cfg.num_eval_episodes, keep_training_loops=True)
        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))
        time.sleep(1)

        torch.save(train_model.state_dict(), f"dqn_agent-{epoch}.pth")

    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
