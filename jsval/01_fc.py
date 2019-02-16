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
    net = nn.Linear(in_features=4, out_features=6)
    print(net)
    print(net.weight)
    print(net.bias)
    print("Probe values:")
    v = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
    print("[0, 0, 0, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    v = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    print("[1, 0, 0, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    v = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
    print("[0, 1, 0, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    v = torch.tensor([[0, 0, 1, 0]], dtype=torch.float32)
    print("[0, 0, 1, 0] -> %s" % net(v)[0].detach().numpy().tolist())
    v = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
    print("[0, 0, 0, 1] -> %s" % net(v)[0].detach().numpy().tolist())

    print("~~~~~~~~~~~~~ JS Network dump")
    w = js.dump_fc(net, 6)
    #print(json.dumps(w, indent=4, sort_keys=True))
    print(f"""
/*###########*/
if (brain) {{
    brain.value_net.fromJSON({json.dumps(w)});
}}
    """)

    pass
