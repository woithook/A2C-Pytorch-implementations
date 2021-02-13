import torch
from torch import nn
import torch.nn.functional as F

# DEVICE = 'cpu'


class Net(nn.Module):
    def __init__(self, l1, l2, l_obs, n_action):
        super(Net, self).__init__()
        self.l_obs = l_obs
        self.n_action = n_action
        self.l1 = l1
        self.l2 = l2

        self.net = nn.Sequential(
            nn.Linear(self.l_obs, self.l1),
            nn.ReLU(),
        )
        self.act = nn.Linear(self.l1, self.n_action)
        self.cri = nn.Linear(self.l1, 1)

    def forward(self, inputs):
        x = self.net(inputs)
        pi_out = self.act(x)
        v_out = self.cri(x)

        return pi_out, v_out


class ConvNet(nn.Module):
    def __init__(self, l1, l2, l_obs, n_action):
        super(ConvNet, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l_obs = l_obs
        self.n_action = n_action
        self.conv = nn.Sequential(
            nn.Conv2d(l_obs, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.ReLU())
        self.act_net = nn.Linear(512, self.n_action)
        self.cri_net = nn.Linear(512, 1)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.act_net(x), self.cri_net(x)


class ConvNet_LSTM(nn.Module):
    def __init__(self, l_obs, n_action, device):
        super(ConvNet_LSTM, self).__init__()
        self.l_obs = l_obs
        self.n_action = n_action
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(l_obs, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(1024, 512, batch_first=True)
        self.act_net = nn.Sequential(
            nn.Linear(512, self.n_action),
            # nn.ReLU(),
            # nn.Linear(128, self.n_action),
        )
        self.cri_net = nn.Sequential(
            nn.Linear(512, 1),
            # nn.ReLU(),
            # nn.Linear(128, 1),
        )
        self.cx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)
        self.hx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(-1, 1024).unsqueeze(0)

        x, (hx, cx) = self.lstm(x, (self.hx, self.cx))
        self.hx = hx
        self.cx = cx

        return self.act_net(x), self.cri_net(x)

    def reset_lstm(self):
        self.cx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)
        self.hx = torch.zeros(512).unsqueeze(0).unsqueeze(0).to(self.device)


class ConvNet_LSTMCell(nn.Module):
    def __init__(self, l_obs, n_action, device):
        super(ConvNet_LSTMCell, self).__init__()
        self.device = device
        self.l_obs = l_obs
        self.n_action = n_action
        self.conv = nn.Sequential(
            nn.Conv2d(l_obs, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTMCell(1024, 512)
        self.act_net = nn.Sequential(
            nn.Linear(512, self.n_action),
            # nn.ReLU(),
            # nn.Linear(128, self.n_action),
        )
        self.cri_net = nn.Sequential(
            nn.Linear(512, 1),
            # nn.ReLU(),
            # nn.Linear(128, 1),
        )
        self.cx = torch.zeros(1, 512).to(self.device)
        self.hx = torch.zeros(1, 512).to(self.device)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (self.hx, self.cx))
        self.hx = hx
        self.cx = cx

        x = hx
        return self.act_net(x), self.cri_net(x)

    def reset_lstm(self):
        self.cx = torch.zeros(1, 512).to(self.device)
        self.hx = torch.zeros(1, 512).to(self.device)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        relu_gain = nn.init.calculate_gain('relu')
        m.weight.data.mul_(relu_gain)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class Trajectory(object):
    def __init__(self):
        self.current_state = []
        self.current_action = []
        self.reward = []
        self.next_state = []

    def add(self, current_state, current_action, reward, next_state=None):
        self.current_state.append(current_state)
        self.current_action.append(current_action)
        self.reward.append(reward)
        if next_state is not None:
            self.next_state.append(next_state)

    def get_state(self):
        return torch.stack(self.current_state, dim=0)

    def get_next_state(self):
        return torch.stack(self.next_state, dim=0)

    def get_reward(self):
        return self.reward

    def get_action(self):
        return self.current_action



