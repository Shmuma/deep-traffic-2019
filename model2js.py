#!/usr/bin/env python3
import argparse
import json

import numpy as np
from libtraffic import env, model, config

import torch.nn as nn


def dump_layers(net, env):
    for m in net.conv.modules():
        d = None
        if isinstance(m, nn.Conv2d):
            # JSConvNet limitations
            assert m.stride[0] == m.stride[1]
            assert m.padding[0] == m.padding[1]
            d = {
                'type': 'conv',
                'sx': m.kernel_size[0],
                'sy': m.kernel_size[1],
                'stride': m.stride[0],
                'pad': m.padding[0],
                'filters': m.out_channels,
                'activation': 'relu',
            }
        elif isinstance(m, nn.MaxPool2d):
            d = {
                'type': 'pool',
                'sx': m.kernel_size,
                'stride': m.stride,
                'pad': m.padding,
            }
        if d is not None:
            print('layer_defs.push(' + json.dumps(d, indent=4, sort_keys=True) + ');')

    for m in net.fc.modules():
        d = None
        if isinstance(m, nn.Linear):
            d = {
                'type': 'fc',
                'num_neurons': m.out_features,
                'activation': 'relu'
            }
            # output layer
            if m.out_features == env.action_space.n:
                d = {
                    'type': 'regression',
                    'num_neurons': env.action_space.n
                }
        if d is not None:
            print('layer_defs.push(' + json.dumps(d, indent=4, sort_keys=True) + ');')


def dump_net(ini, net, env, state_dict):
    # params first
    print(f"""
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = {ini.env_lanes_side};
patchesAhead = {ini.env_patches_ahead};
patchesBehind = {ini.env_patches_ahead};
trainIterations = 10000;

// the number of other autonomous vehicles controlled by your network
otherAgents = 0; // max of 10

//var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = {ini.env_history};
//var exp_network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;
//var network_size = num_inputs * (temporal_window + num_actions*temporal_window + 1);
var input_x = lanesSide*2+1;
var input_y = patchesAhead + patchesBehind;

var layer_defs = [];
layer_defs.push({{
    type: 'input',
    out_sx: input_x,
    out_sy: input_y,
    out_depth: temporal_window + num_actions*temporal_window + 1
}});

""")
    dump_layers(net, env)

    print(f"""
var tdtrainer_options = {{
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
}};

var opt = {{}};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {{
    brain.backward(lastReward);
    var action = brain.forward(state);
    draw_net();
    draw_stats();
    return action;
}}

//]]>
    """)
    pass



if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=160)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', required=True, help="Model to load")
    parser.add_argument("-i", "--ini", required=True, help="Ini file to use params")
    args = parser.parse_args()
    ini = config.Settings(args.ini)

    e = env.DeepTraffic(lanes_side=ini.env_lanes_side, patches_ahead=ini.env_patches_ahead,
                        patches_behind=ini.env_patches_behind, history=ini.env_history)
    net = model.DQN(e.obs_shape, e.action_space.n)
#    net.load_state_dict()
#    state_dict = torch.load(args.model)
    state_dict = {}

    dump_net(ini, net, e, state_dict)
