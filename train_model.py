#!/usr/bin/env python3
import argparse
import logging

import ptan
import pathlib
import numpy as np
import gym.wrappers
from tensorboardX import SummaryWriter
from libtraffic import env, utils, model

import torch
import torch.optim as optim

log = logging.getLogger("train_model")

# obs hyperparams
LANES_SIDE = 1
PATCHES_AHEAD = 20
PATCHES_BEHIND = 10
HISTORY = 3

ENV_STEPS_LIMIT = 100
EPS_START = 1.0
EPS_END = 0.05
EPS_STEPS = 100000

GAMMA = 0.95
REPLAY_SIZE = 100000
MIN_REPLAY = 1000
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
NET_SYNC_STEPS = 250
TEST_STEPS = 1000
L2_REG = 1e-3


def test_agent(net, steps=1000, rounds=5, device=torch.device('cpu')):
    round_means = []
    for _ in range(rounds):
        speed_hist = []
        test_env = env.DeepTraffic(lanes_side=LANES_SIDE, patches_ahead=PATCHES_AHEAD,
                                   patches_behind=PATCHES_BEHIND, history=HISTORY)
        obs = test_env.reset()

        for _ in range(steps):
            speed_hist.append(test_env.current_speed())
            obs_v = torch.tensor([obs]).to(device)
            q_v = net(obs_v)[0]
            act_idx = torch.argmax(q_v).item()
            obs, reward, _, _ = test_env.step(act_idx)
        round_means.append(np.mean(speed_hist))
    return np.mean(round_means), np.std(round_means)


if __name__ == "__main__":
    utils.setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda calculations")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = pathlib.Path("saves") / args.name
    save_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(comment="-" + args.name)

    e = env.DeepTraffic(lanes_side=LANES_SIDE, patches_ahead=PATCHES_AHEAD,
                        patches_behind=PATCHES_BEHIND, history=HISTORY)
    obs_shape = e.obs_shape
    e = gym.wrappers.TimeLimit(e, max_episode_steps=ENV_STEPS_LIMIT)

    log.info("Environment created, obs shape %s", obs_shape)
    net = model.DQN(obs_shape, e.action_space.n).to(device)
    log.info("Model: %s", net)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPS_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(e, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    step = 0
    losses = []
    best_test_speed = None
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
            loss_v = model.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=GAMMA, device=device)
            loss_v.backward()
            losses.append(loss_v.item())
            optimizer.step()

            if step % NET_SYNC_STEPS == 0:
                tgt_net.sync()
            if step % TEST_STEPS == 0:
                mean_loss = np.mean(losses)
                losses.clear()
                car_speed_mu, car_speed_std = test_agent(net, device=device)
                log.info("%d: loss=%.3f, test_speed_mu=%.2f, test_speed_std=%.2f", step, mean_loss, car_speed_mu,
                         car_speed_std)
                writer.add_scalar("loss", mean_loss, step)
                writer.add_scalar("car_speed_mu", car_speed_mu, step)
                writer.add_scalar("car_speed_std", car_speed_std, step)

                if best_test_speed is None:
                    best_test_speed = car_speed_mu
                elif best_test_speed < car_speed_mu:
                    log.info("Best speed updated: %.2f -> %.2f", best_test_speed, car_speed_mu)
                    best_test_speed = car_speed_mu
                    torch.save(net.state_dict(), save_path / ("best_%.2f.dat" % best_test_speed))
    pass
