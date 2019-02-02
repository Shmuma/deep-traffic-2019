#!/usr/bin/env python3
import argparse
import json

import numpy as np
from libtraffic import env, model, config

import torch
import torch.nn as nn


def np_to_dict(arr):
    shape = arr.shape
    if len(shape) == 3:
        sx, sy, depth = shape
    else:
        sx, sy = 1, 1
        depth = shape[0]
    return {
        "sx": sx,
        "sy": sy,
        "depth": depth,
        "w": {str(idx): float(val) for idx, val in enumerate(arr.flatten())}
    }


def conv_weights(m):
    assert isinstance(m, nn.Conv2d)

    res = {
        'layer_type': 'conv',
        'sx': m.kernel_size[0],
        'sy': m.kernel_size[1],
        'in_depth': m.in_channels,
        'out_depth': m.out_channels,
        'stride': m.stride[0],
        'pad': m.padding[0],
    }

    sdict = m.state_dict()
    w = sdict['weight']
    res_filters = []
    for filt_idx in range(w.size()[0]):
        filt = w[filt_idx]
        filt = filt.permute(1, 2, 0)
        f_dict = np_to_dict(filt.numpy())
        res_filters.append(f_dict)
    res['filters'] = res_filters

    b = sdict['bias']
    res['biases'] = np_to_dict(b.numpy())
    return res


def relu_weights(m, size):
    assert isinstance(m, nn.ReLU)

    return {
        "out_depth": size,
        "out_sx": 1,
        "out_sy": 1,
        "layer_type": "relu"
    }


def pool_weights(m, size):
    assert isinstance(m, nn.MaxPool2d)

    return {
        'layer_type': 'pool',
        'sx': m.kernel_size,
        'stride': m.stride,
        'pad': m.padding,
        "in_depth": size,
        "out_depth": size,
        "out_sx": 1,
        "out_sy": ','
    }


def dump_layers(net, env):
    layers = []
    weights = []
    conv_size = None
    for m in net.conv.modules():
        d = None
        w = None
        if isinstance(m, nn.Conv2d):
            # JSConvNet limitations
            assert m.stride[0] == m.stride[1]
            assert m.padding[0] == m.padding[1]
            d = {
                'type': 'conv',
                'sx': m.kernel_size[0],
                'sy': m.kernel_size[1],
                'in_depth': m.in_channels,
                'stride': m.stride[0],
                'pad': m.padding[0],
                'filters': m.out_channels,
                'activation': 'relu',
            }
            w = conv_weights(m)
            conv_size = m.out_channels
        elif isinstance(m, nn.ReLU):
            w = relu_weights(m, conv_size)
        elif isinstance(m, nn.MaxPool2d):
            d = {
                'type': 'pool',
                'sx': m.kernel_size,
                'stride': m.stride,
                'pad': m.padding,
            }
            w = pool_weights(m, conv_size)
        if d is not None:
            print('layer_defs.push(' + json.dumps(d, indent=4, sort_keys=True) + ');')
        if w is not None:
            weights.append(w)

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


# TODO: fix input layer shape and dump
# TODO: dump of FC weights

def dump_net(ini, net, env):
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
                        patches_behind=ini.env_patches_behind, history=ini.env_history, obs=ini.env_obs)
    model_class = model.MODELS[ini.train_model]
    net = model_class(e.obs_shape, e.action_space.n)
    print(net)
    net.load_state_dict(torch.load(args.model))

    dump_net(ini, net, e)
