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
    reg_net = nn.Sequential(
        nn.Linear(in_features=5*1*1, out_features=6)
    )
    full_net = lambda x: reg_net(net(x).view(x.size()[0], -1))
    print(net)
    print(reg_net)
    print("Probe values:")
    z = torch.zeros(1, 2, 3, 3)
    r = full_net(z)
    print("zero: %s -> %s" % (z.size(), r.detach().numpy()[0]))
    for x in range(3):
        for y in range(3):
            z[0, 0, x, y] = 1.0
            r = full_net(z)
            print("(0, %d, %d) = 1 -> %s" % (x, y, r.detach().numpy()[0]))
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
    f_layers, f_weights = js.fc(reg_net)
    layers.extend(f_layers)
    weights.extend(f_weights)
    js.add_input_output(layers, weights, in_shape=(2, 3, 3), out_shape=6)
    js.print_layers(layers)
    print(f"""
    /*###########*/
    if (brain) {{
        brain.value_net.fromJSON({json.dumps({"layers": weights})});
    }}
        """)
