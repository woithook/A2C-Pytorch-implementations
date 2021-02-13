"""
implementation of A2C with Pytorch for the atari game Breakout

version: 0.4
Use only one trajectory for training once.
"""

import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import model
from model_ac import Net, ConvNet_LSTM, ConvNet_LSTMCell
from model import Trajectory
from utils import atari_env


LEARNING_RATE = 0.0001
GAMMA = 0.99
NUM_ITER = 50000

env_name = 'PongDeterministic-v4'
use_epsilon_greedy = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'

seed = 123456789
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def main():
    if use_epsilon_greedy:
        epsilon = 0.5
        epsilon_end = 0.1
        epsilon_div = 1e3
        epsilon_step = (
                (epsilon - epsilon_end) / epsilon_div)
    else:
        epsilon = 0.0

    env = atari_env(env_name)
    l_obs = env.observation_space.shape[0]
    n_action = env.action_space.n

    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dir = 'runs/Breakout_a2c_v2_experiment_epoch5000/' + date
    writer = SummaryWriter(dir)

    net = ConvNet_LSTMCell(l_obs, n_action, device=DEVICE).to(DEVICE)
    net.apply(model.weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    results_reward = []

    for i_iteration in range(NUM_ITER):
        net.reset_lstm()

        obs = env.reset()
        obs = torch.Tensor(obs).unsqueeze(0)

        total_reward = 0
        done = False
        traj = Trajectory()

        while not done:
            action = action_decide(net, obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            traj.add(obs, action, reward)
            total_reward += reward

            obs = next_obs
            obs = torch.Tensor(obs).unsqueeze(0)
            # if i_iteration >= 0:
            #     env.render()
            #     time.sleep(1/60)
            if done:
                results_reward.append(total_reward)
                writer.add_scalar("Reward/epoch", total_reward, i_iteration + 1)
        print('iteration: ', i_iteration + 1, '/ ', NUM_ITER, ' reward: ', total_reward)
        A2C(net, optimizer, traj)
        torch.cuda.empty_cache()
        if use_epsilon_greedy:
            if epsilon > epsilon_end:
                epsilon -= epsilon_step
            else:
                epsilon = epsilon_end
    env.close()
    writer.flush()
    writer.close()
    return results_reward


def A2C(net, optimizer, trajectory):
    states = trajectory.get_state()
    actions = trajectory.get_action()
    returns = trajectory.get_returns(GAMMA)

    net.reset_lstm()
    optimizer.zero_grad()

    states = states.to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(DEVICE)
    return_ = torch.Tensor(returns).to(DEVICE)

    logits = torch.zeros(len(states), net.n_action).to(DEVICE)
    v = torch.zeros(len(states)).to(DEVICE)
    for i in range(len(states)):
        logits_p, v_p = net(states[i])
        logits[i] = logits_p.view(-1)
        v[i] = v_p.view(-1)

    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    log_prob_act = log_prob.gather(1, actions).view(-1)

    q = (return_ + return_.mean()) / (return_.std() + torch.finfo(torch.float32).eps).detach()
    a = q - v
    a = (a - a.mean()) / (a.std() + torch.finfo(torch.float32).eps)

    loss_policy = - (a * log_prob_act).sum()
    loss_critic = a.pow(2).sum()
    # loss_critic = F.smooth_l1_loss(q, v).sum()
    loss_entropy = - (log_prob * prob).sum()

    loss = loss_policy + 0.5 * loss_critic + 0.01 * loss_entropy
    # loss = loss_policy + 0.5 * loss_critic

    loss.backward()

    # total_norm = 0
    # for p in net.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
    optimizer.step()


def action_decide(net, obs, epsilon=0.0):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        logits, _ = net(obs.to(DEVICE))
        logits = logits.view(-1)
        probs = F.softmax(logits, dim=0)
        probs_hat = (1 - epsilon) * probs + epsilon / net.n_action

        m = Categorical(probs_hat)
        action = m.sample()
        return action.item()


if __name__ == '__main__':
    results = main()
    plt.title("learning rate = %f" % LEARNING_RATE)
    plt.plot(results)
    plt.show()
