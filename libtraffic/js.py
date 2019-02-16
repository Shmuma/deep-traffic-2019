import json
import torch.nn as nn


def norm_float(v):
    if abs(v) < 1e-20:
        return 0.0
    return v


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
        "w": {str(idx): norm_float(float(val)) for idx, val in enumerate(arr.flatten())}
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
        "l1_decay_mul": 0,
        "l2_decay_mul": 1,
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


def fc_weights(m):
    assert isinstance(m, nn.Linear)

    res = {
        "out_sx": 1,
        "out_sy": 1,
        "out_depth": m.out_features,
        "num_inputs": m.in_features,
        "layer_type": "fc",
        "l1_decay_mul": 0,
        "l2_decay_mul": 1,
    }

    sdict = m.state_dict()
    w = sdict['weight']
    res_filters = []
    for filt_idx in range(w.size()[0]):
        filt = w[filt_idx]
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


def dump_fc(net, output_size):
    weights = []
    input_created = False
    prev_out_size = None

    for m in net.modules():
        d = None
        w = None

        if isinstance(m, nn.Linear):
            prev_out_size = m.out_features
            d = {
                'type': 'fc',
                'num_neurons': m.out_features,
                'activation': 'relu',
            }
            w = fc_weights(m)
            # output layer
            if m.out_features == output_size:
                d = {
                    'type': 'regression',
                    'num_neurons': output_size
                }
        elif isinstance(m, nn.ReLU):
            w = {"out_depth": prev_out_size, "out_sx": 1, "out_sy": 1, "layer_type": "relu"}
        if d is not None:
            print('layer_defs.push(' + json.dumps(d, indent=4, sort_keys=True) + ');')
        if w is not None:
            if not input_created:
                weights.append({
                    "out_depth": m.in_features,
                    "out_sx": 1,
                    "out_sy": 1,
                    "layer_type": "input"
                })
                input_created = True
            weights.append(w)
    weights.append({
        "out_depth": output_size,
        "out_sx": 1,
        "out_sy": 1,
        "layer_type": "regression",
        "num_inputs": output_size,
    })

    return {
        "layers": weights
    }




def dump_layers(net, output_size):
    weights = []
    input_created = False
    if hasattr(net, 'conv'):
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
        w = None
        if isinstance(m, nn.Linear):
            d = {
                'type': 'fc',
                'num_neurons': m.out_features,
                'activation': 'relu',
            }
            w = fc_weights(m)
            # output layer
            if m.out_features == output_size:
                d = {
                    'type': 'regression',
                    'num_neurons': output_size
                }
        if d is not None:
            print('layer_defs.push(' + json.dumps(d, indent=4, sort_keys=True) + ');')
        if w is not None:
            if not input_created:
                weights.append({
                    "out_depth": m.in_features,
                    "out_sx": 1,
                    "out_sy": 1,
                    "layer_type": "input"
                })
                input_created = True
            weights.append(w)
    weights.append({
        "out_depth": output_size,
        "out_sx": 1,
        "out_sy": 1,
        "layer_type": "regression",
        "num_inputs": output_size,
    })

    return {
        "layers": weights
    }
