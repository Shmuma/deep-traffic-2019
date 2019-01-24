#!/usr/bin/env python3
import argparse
import logging

import pathlib
import numpy as np
from libtraffic import env, utils, model, config

import torch

log = logging.getLogger("model_play")

# obs hyperparams
LANES_SIDE = 1
PATCHES_AHEAD = 20
PATCHES_BEHIND = 10
HISTORY = 3


def play_episode(e, net, steps=1000, verbose=False):
    obs = e.reset()
    speed_hist = []

    for _ in range(steps):
        speed_hist.append(e.current_speed())
        obs_v = torch.tensor([obs])
        q_v = net(obs_v)[0]
        act_idx = torch.argmax(q_v).item()

        occ = e.state.render_occupancy(full=False)
        if verbose:
            print(env.Actions(act_idx), e.state.my_car.safe_speed)
            print(occ)

        obs, reward, _, _ = e.step(act_idx)
        if verbose:
            print(reward)
    return np.mean(speed_hist)


if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=160)
    utils.setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', required=True, help="Model to load")
    parser.add_argument("-i", "--ini", required=True, help="Ini file to use params")
    parser.add_argument("-v", "--verbose", action='store_true', help="Dispaly individual steps")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Limit of steps, default=100")
    args = parser.parse_args()
    ini = config.Settings(args.ini)

    e = env.DeepTraffic(lanes_side=ini.env_lanes_side, patches_ahead=ini.env_patches_ahead,
                        patches_behind=ini.env_patches_behind, history=ini.env_history)
    obs_shape = e.obs_shape
    net = model.DQN(obs_shape, e.action_space.n)
    net.load_state_dict(torch.load(args.model))

    log.info("Model loaded from %s", args.model)
    mean_speed = play_episode(e, net, steps=args.steps, verbose=args.verbose)
    log.info("Mean speed is %.3f", mean_speed)

