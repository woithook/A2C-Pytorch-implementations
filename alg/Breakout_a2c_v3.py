"""
implementation of A2C with Pytorch

version: 0.3
Based on a2c_v2.
Use only one trajectory for training once.
"""

import os
import time
import random
import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import model
from model_ac import Net, ConvNet_LSTM
from model import Trajectory
from utils import process_frame
from utils import atari_env


LEARNING_RATE = 0.0001
GAMMA = 0.99
NUM_ITER = 50000

env_name = 'Breakout-v0'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'

seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def main():
    epsilon = 0.5
    epsilon_end = 0.01
    epsilon_div = 1e4
    epsilon_step = (
            (epsilon - epsilon_end) / epsilon_div)

    env = atari_env(env_name)
    l_obs = env.observation_space.shape[0]
    n_action = env.action_space.n

    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dir = 'runs/Breakout_a2c_v2_experiment_epoch5000/' + date
    writer = SummaryWriter(dir)

    net = ConvNet_LSTM(l_obs, n_action).to(DEVICE)
    net.apply(model.weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    results_reward = []

    for i_iteration in range(NUM_ITER):
        buff = []

        net.reset_lstm()

        obs = env.reset()
        obs = torch.Tensor(obs).unsqueeze(0)
        # obs = torch.Tensor(obs)

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
            # if i_episode == 0:
            #     env.render()
            #     time.sleep(0.03)
            if done:
                results_reward.append(total_reward)
                writer.add_scalar("Reward/epoch", total_reward, i_iteration + 1)
        print('iteration: ', i_iteration + 1, '/ ', NUM_ITER, ' reward: ', total_reward)
        A2C(net, optimizer, traj)
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

    states = states.to(DEVICE).view(-1, 1, 80, 80)
    return_ = torch.Tensor(returns).detach().to(DEVICE)

    logits, v = net(states)
    logits = logits.view(-1, net.n_action)
    v = v.view(-1)
    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    log_prob_act = torch.stack([real_log[real_act] for real_log, real_act in zip(log_prob, actions)])

    q = return_
    a = q - v
    a = (a - a.mean()) / (a.std() + torch.finfo(torch.float32).eps)

    loss_policy = - torch.dot(a, log_prob_act).view(1)
    loss_critic = a.pow(2).mean()
    loss_entropy = - torch.dot(log_prob.view(-1), prob.view(-1)) / len(logits)

    loss = loss_policy + 0.5 * loss_critic + 0.01 * loss_entropy
    # loss = loss_policy + 0.5 * loss_critic

    loss.backward()

    # total_norm = 0
    # for p in net.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)
    # # torch.nn.utils.clip_grad_norm_(net.parameters(), 30)
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
