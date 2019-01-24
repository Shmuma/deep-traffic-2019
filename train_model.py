#!/usr/bin/env python3
import argparse
import logging

import ptan
import numpy as np
import gym.wrappers
from tensorboardX import SummaryWriter
from libtraffic import env, utils, model

import torch.optim as optim

log = logging.getLogger("train_model")

# obs hyperparams
LANES_SIDE = 1
PATCHES_AHEAD = 20
PATCHES_BEHIND = 10
HISTORY = 3

ENV_STEPS_LIMIT = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_STEPS = 10000

GAMMA = 0.95
REPLAY_SIZE = 1000
MIN_REPLAY = 256
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NET_SYNC_STEPS = 128


if __name__ == "__main__":
    utils.setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    writer = SummaryWriter(comment="-" + args.name)

    e = env.DeepTraffic(lanes_side=LANES_SIDE, patches_ahead=PATCHES_AHEAD,
                        patches_behind=PATCHES_BEHIND, history=HISTORY)
    obs_shape = e.obs_shape
    e = gym.wrappers.TimeLimit(e, max_episode_steps=ENV_STEPS_LIMIT)

    log.info("Environment created, obs shape %s", obs_shape)
    net = model.DQN(obs_shape, e.action_space.n)
    log.info("Model: %s", net)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPS_START)
    agent = ptan.agent.DQNAgent(net, selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(e, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    step = 0
    losses = []
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            buffer.populate(1)
            if len(buffer) < MIN_REPLAY:
                continue
            step += 1
            selector.epsilon = max(EPS_END, EPS_START - step / EPS_STEPS)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                tracker.reward(new_rewards[-1], step, epsilon=selector.epsilon)
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = model.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=GAMMA)
            loss_v.backward()
            losses.append(loss_v.item())
            optimizer.step()

            if step % NET_SYNC_STEPS == 0:
                tgt_net.sync()
            if len(losses) >= 100:
                mean_loss = np.mean(losses)
                log.info("%d: loss=%.3f", step, mean_loss)
                losses.clear()
    pass
