import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=(1, 0)),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(obs_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        o = self.conv(x)
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
