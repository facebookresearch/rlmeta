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

import rlmeta.envs.atari_wrapper as atari_wrapper
import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.random_utils as random_utils
import rlmeta.utils.remote_utils as remote_utils

from examples.atari.dqn.atari_dqn_model import AtariDQNModel
from rlmeta.agents.dqn import (ApexDQNAgent, ApexDQNAgentFactory,
                               ConstantEpsFunc, FlexibleEpsFunc)
from rlmeta.core.controller import Phase, Controller
from rlmeta.core.loop import LoopList, ParallelLoop
from rlmeta.core.model import ModelVersion, RemotableModelPool
from rlmeta.core.model import make_remote_model, wrap_downstream_model
from rlmeta.core.replay_buffer import ReplayBuffer, make_remote_replay_buffer
from rlmeta.core.server import Server, ServerList
from rlmeta.samplers import PrioritizedSampler
from rlmeta.storage import TensorCircularBuffer
from rlmeta.utils.optimizer_utils import make_optimizer


@hydra.main(config_path="./conf", config_name="conf_apex_dqn")
def main(cfg):
    if cfg.seed is not None:
        random_utils.manual_seed(cfg.seed)
    logging.info(hydra_utils.config_to_json(cfg))

    env = atari_wrapper.make_atari_env(**cfg.env)
    model = AtariDQNModel(env.action_space.n,
                          double_dqn=cfg.double_dqn).to(cfg.train_device)
    model_pool = RemotableModelPool(copy.deepcopy(model).to(cfg.infer_device),
                                    seed=cfg.seed)
    optimizer = make_optimizer(model.parameters(), **cfg.optimizer)

    replay_buffer = ReplayBuffer(
        TensorCircularBuffer(cfg.replay_buffer_size),
        PrioritizedSampler(priority_exponent=cfg.priority_exponent))
    ctrl = Controller()

    m_server = Server(cfg.m_server_name, cfg.m_server_addr)
    r_server = Server(cfg.r_server_name, cfg.r_server_addr)
    c_server = Server(cfg.c_server_name, cfg.c_server_addr)
    m_server.add_service(model_pool)
    r_server.add_service(replay_buffer)
    c_server.add_service(ctrl)
    servers = ServerList([m_server, r_server, c_server])

    learner_model = wrap_downstream_model(model, m_server)
    t_actor_model = make_remote_model(model,
                                      m_server,
                                      version=ModelVersion.LATEST)
    # During blocking evaluation we have STABLE is LATEST
    e_actor_model = make_remote_model(model,
                                      m_server,
                                      version=ModelVersion.LATEST)

    learner_ctrl = remote_utils.make_remote(ctrl, c_server)
    t_actor_ctrl = remote_utils.make_remote(ctrl, c_server)
    e_actor_ctrl = remote_utils.make_remote(ctrl, c_server)

    learner_replay_buffer = make_remote_replay_buffer(replay_buffer,
                                                      r_server,
                                                      prefetch=cfg.prefetch)
    t_actor_replay_buffer = make_remote_replay_buffer(replay_buffer, r_server)

    env_fac = atari_wrapper.AtariWrapperFactory(**cfg.env)

    t_agent_fac = ApexDQNAgentFactory(t_actor_model,
                                      FlexibleEpsFunc(
                                          cfg.eps, cfg.num_training_rollouts),
                                      replay_buffer=t_actor_replay_buffer,
                                      n_step=cfg.n_step,
                                      max_abs_reward=cfg.max_abs_reward,
                                      rescale_value=cfg.rescale_value)
    e_agent_fac = ApexDQNAgentFactory(e_actor_model,
                                      ConstantEpsFunc(cfg.evaluation_eps))

    t_loop = ParallelLoop(env_fac,
                          t_agent_fac,
                          t_actor_ctrl,
                          running_phase=Phase.TRAIN,
                          should_update=True,
                          num_rollouts=cfg.num_training_rollouts,
                          num_workers=cfg.num_training_workers,
                          seed=cfg.seed)
    e_loop = ParallelLoop(env_fac,
                          e_agent_fac,
                          e_actor_ctrl,
                          running_phase=Phase.EVAL,
                          should_update=False,
                          num_rollouts=cfg.num_evaluation_rollouts,
                          num_workers=cfg.num_evaluation_workers,
                          seed=(None if cfg.seed is None else cfg.seed +
                                cfg.num_training_rollouts))
    loops = LoopList([t_loop, e_loop])

    learner = ApexDQNAgent(
        learner_model,
        replay_buffer=learner_replay_buffer,
        controller=learner_ctrl,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        n_step=cfg.n_step,
        importance_sampling_exponent=cfg.importance_sampling_exponent,
        value_clipping_eps=cfg.value_clipping_eps,
        target_sync_period=cfg.target_sync_period,
        learning_starts=cfg.learning_starts,
        model_push_period=cfg.model_push_period)

    servers.start()
    loops.start()
    learner.connect()

    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        stats = learner.train(cfg.steps_per_epoch)
        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))
        time.sleep(1)

        stats = learner.eval(cfg.num_evaluation_episodes,
                             keep_training_loops=True)
        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))

        torch.save(model.state_dict(), f"dqn_agent-{epoch}.pth")
        time.sleep(1)

    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
