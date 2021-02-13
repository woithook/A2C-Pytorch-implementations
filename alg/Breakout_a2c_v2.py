"""
implementation of A2C with Pytorch

version: 0.2
Based on a2c_v2
Adding LSTM layer to the network.
The code work but the performance is low.
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


LEARNING_RATE = 0.0001
GAMMA = 0.99
NUM_ITER = 50000
NUM_EPISODE = 1

env_name = 'Breakout-v0'
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def main():
    env = gym.make(env_name)
    l_obs = 1
    n_action = env.action_space.n

    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dir = 'runs/Breakout_a2c_v2_experiment_epoch5000/' + date
    writer = SummaryWriter(dir)

    net = ConvNet_LSTM(l_obs, n_action, device=DEVICE).to(DEVICE)
    net.apply(model.weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    results_reward = []

    for i_iteration in range(NUM_ITER):
        buff = []
        avg_reward = 0

        for i_episode in range(NUM_EPISODE):
            net.reset_lstm()

            obs = env.reset()
            obs = process_frame(obs)
            obs = torch.Tensor(obs).unsqueeze(0)

            total_reward = 0
            done = False
            traj = Trajectory()

            while not done:
                action = action_decide(net, obs)
                next_obs, reward, done, _ = env.step(action)
                traj.add(obs, action, reward)
                total_reward += reward

                obs = next_obs
                obs = process_frame(obs)
                obs = torch.Tensor(obs).unsqueeze(0)
                # if i_episode == 0:
                #     env.render()
                #     time.sleep(0.03)
                if done:
                    buff.append(traj)
                    results_reward.append(total_reward)
                    avg_reward += total_reward
                    writer.add_scalar("Reward/epoch", total_reward, i_iteration * NUM_EPISODE + (i_episode + 1))
        print('iteration: ', i_iteration + 1, '/ ', NUM_ITER, ' average reward: ', avg_reward / NUM_EPISODE)
        A2C(net, optimizer, buff)
    env.close()
    writer.flush()
    writer.close()
    return results_reward


def A2C(net, optimizer, buff):
    optimizer.zero_grad()

    states = [trajectory.get_state() for trajectory in buff]
    actions = [trajectory.get_action() for trajectory in buff]
    returns = [trajectory.get_returns(GAMMA) for trajectory in buff]
    loss_policy = []
    loss_critic = []
    loss_entropy = []

    for i in range(len(buff)):
        net.reset_lstm()

        state = states[i].to(DEVICE).view(-1, 1, 80, 80)
        action = actions[i]
        return_ = torch.Tensor(returns[i]).detach().to(DEVICE)

        logits, v = net(state)
        logits = logits.view(-1, net.n_action)
        v = v.view(-1)
        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_act = torch.stack([real_log[real_act] for real_log, real_act in zip(log_prob, action)])

        q = return_
        a = q - v
        a = (a - a.mean()) / (a.std() + torch.finfo(torch.float32).eps)

        loss_policy_p = - torch.dot(a, log_prob_act).view(1) / len(logits)
        loss_policy.append(loss_policy_p)

        loss_critic_p = a.pow(2).mean()
        loss_critic.append(loss_critic_p)

        loss_entropy_p = - torch.dot(log_prob.view(-1), prob.view(-1)) / len(logits)
        loss_entropy.append(loss_entropy_p)

    loss_policy = torch.stack(loss_policy).mean()
    loss_critic = torch.stack(loss_critic).mean()
    loss_entropy = torch.stack(loss_entropy).mean()
    loss = loss_policy + 0.5 * loss_critic + 0.01 * loss_entropy
    # loss = loss_policy + 0.5 * loss_critic

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.36)
    optimizer.step()

    torch.cuda.empty_cache()


def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        logits, _ = net(obs.to(DEVICE))
        logits = logits.view(-1)
        probs = F.softmax(logits, dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


if __name__ == '__main__':
    results = main()
    plt.title("learning rate = %f" % LEARNING_RATE)
    plt.plot(results)
    plt.show()
