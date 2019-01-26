Model

```
2019-01-25 15:44:15 INFO Model: DQN(
  (conv): Sequential(
    (0): Conv2d(19, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=256, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=5, bias=True)
  )
)
```


Ini file
```ini
[env]
lanes_side=3
patches_ahead=20
patches_behind=10
histary=3
steps_limit=200

[train]
gamma=0.95
lr=1e-4
replay_size=1000000
replay_initial=1000
cuda=True
eps_start = 1.0
eps_end = 0.15
eps_steps = 100000
batch_size=512
l2_reg=1e-3
# one step every 1000 steps
add_steps_limit_slope=0.002
add_steps_limit_max=1800
net_sync_steps=10000
```

Java script:
```javascript
lanesSide = 3;
patchesAhead = 20;
patchesBehind = 10;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;

var temporal_window = 3;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;
```
