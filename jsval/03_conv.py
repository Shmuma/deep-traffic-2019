# simple 1-layer FC net
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

import json
import torch
import torch.nn as nn

from libtraffic import js


if __name__ == "__main__":
    torch.random.manual_seed(123)
    net = nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=5, kernel_size=3),
        nn.ReLU(),
    )
    print(net)
    # it = net.modules()
    # next(it)
    # l = next(it)
    # print(l.weight)
    # print(l.bias)
    print("Probe values:")
    z = torch.zeros(1, 2, 5, 5)
    r = net(z)
    print("zero: %s ->\n%s" % (z.size(), r))
    for x in range(5):
        for y in range(5):
            z[0, 0, x, y] = 1.0
            r = net(z)
            print("(0, %d, %d) = 1 ->\n%s" % (x, y, r))
            z[0, 0, x, y] = 0.0
    # print("[0, 0, 0, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    # v = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    # print("[1, 0, 0, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    # v = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
    # print("[0, 1, 0, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    # v = torch.tensor([[0, 0, 1, 0]], dtype=torch.float32)
    # print("[0, 0, 1, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    # v = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
    # print("[0, 0, 0, 1] -> %s" % net(v)[0].detach().numpy().tolist())
    #
    print("~~~~~~~~~~~~~ JS Network dump")
    layers, weights = js.conv(net)
    js.add_input_output(layers, weights, in_shape=(2, 5, 5), out_shape=(5, 3, 3))
    js.print_layers(layers)
    print(f"""
    /*###########*/
    if (brain) {{
        brain.value_net.fromJSON({json.dumps({"layers": weights})});
    }}
        """)
