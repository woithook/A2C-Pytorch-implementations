import numpy as np
import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaLinear, MetaSequential)
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, l1, l2, l_obs, n_action):
        super(Net, self).__init__()
        self.l1 = nn.Linear(l_obs, l1)
        self.l2 = nn.Linear(l1, l2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(l2, n_action)

    def forward(self, input):
        z1 = self.relu1(self.l1(input))
        z2 = self.relu2(self.l2(z1))
        output = self.l3(z2)
        return output


class Net_LSTM(nn.Module):
    def __init__(self, l1, l2, l_obs, n_action):
        super(Net_LSTM, self).__init__()
        self.l_obs = l_obs
        self.n_action = n_action
        self.l1 = l1
        self.l2 = l2
        self.hx = torch.zeros(self.l1).unsqueeze(0).unsqueeze(0)
        self.cx = torch.zeros(self.l1).unsqueeze(0).unsqueeze(0)

        self.lstm = nn.LSTM(l_obs, l1, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(l1, n_action),
            # nn.ReLU(),
            # nn.Linear(l2, n_action),
        )

    def forward(self, inputs):
        x, (hx, cx) = self.lstm(inputs, (self.hx, self.cx))
        self.hx = hx
        self.cx = cx

        outputs = self.net(x)
        return outputs

    def reset_lstm(self):
        self.hx = torch.zeros(self.l1).unsqueeze(0).unsqueeze(0)
        self.cx = torch.zeros(self.l1).unsqueeze(0).unsqueeze(0)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class MetaNet(MetaModule):
    def __init__(self, l1, l2, l_obs, n_action):
        super(MetaNet, self).__init__()
        self.l_obs = l_obs
        self.n_action = n_action
        self.l1 = l1
        self.l2 = l2
        self.actor_net = MetaSequential(
            MetaLinear(self.l_obs, self.l1),
            nn.ReLU(),
            MetaLinear(self.l1, self.l2),
            nn.ReLU(),
            MetaLinear(self.l2, self.n_action),
        )

    def forward(self, inputs, params=None):
        pi_out = self.actor_net(inputs, params=self.get_subdict(params, 'actor_net'))
        return pi_out


def weight_init_meta(m):
    if type(m) == MetaLinear:
        # nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.xavier_normal_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Trajectory(object):
    def __init__(self):
        self.current_state = []
        self.current_action = []
        self.reward = []

    def add(self, current_state, current_action, reward):
        self.current_state.append(current_state)
        self.current_action.append(current_action)
        self.reward.append(reward)

    def get_state(self):
        return torch.stack(self.current_state, dim=0)

    def get_reward(self):
        return self.reward

    def get_returns(self, gamma=0.9):
        R = 0
        returns = []
        for r in self.reward[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        # returns = (returns - np.mean(returns)) / (np.std(returns) + np.finfo(np.float32).eps)
        return returns

    def get_action(self):
        return self.current_action



