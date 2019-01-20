#!/usr/bin/env python3
import argparse
import logging
import pathlib

import ptan
import gym.wrappers
from tensorboardX import SummaryWriter
from libtraffic import env, utils, model

log = logging.getLogger("train_model")

# obs hyperparams
LANES_SIDE = 1
PATCHES_AHEAD = 20
PATCHES_BEHIND = 10
HISTORY = 3
ENV_STEPS_LIMIT = 1000


if __name__ == "__main__":
    utils.setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    e = env.DeepTraffic(lanes_side=LANES_SIDE, patches_ahead=PATCHES_AHEAD,
                        patches_behind=PATCHES_BEHIND, history=HISTORY)
    obs_shape = e.obs_shape
    e = gym.wrappers.TimeLimit(e, max_episode_steps=ENV_STEPS_LIMIT)

    log.info("Environment created, obs shape %s", obs_shape)
    net = model.DQN(obs_shape, e.action_space.n)
    log.info("Model: %s", net)

    pass
