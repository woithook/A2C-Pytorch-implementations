"""
Implement of policy gradient in Cartpole environment
Use cumulative reward R as f(s,a) (i.e. REINFORCE algorithm)

version 0.6:
Based on v0.2
Remove velocity dim in the observation to make CartPole be a POMDP.
Then try to use lstm to train agent.

Tuning
"""

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
from model import Net, Net_LSTM
from model import Trajectory

use_lstm = True

LEARNING_RATE = 0.005
GAMMA = 0.999
NUM_ITER = 400
NUM_EPISODE = 25
eps = torch.finfo(torch.float32).eps

env_name = 'CartPole-v1'

seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def main():
    env = gym.make(env_name)
    l_obs = env.observation_space.shape[0] - 1
    n_action = env.action_space.n

    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dir = 'runs/cartpole_pg_v6_experiment_epoch5000/' + date
    writer = SummaryWriter(dir)

    if use_lstm:
        net = Net_LSTM(64, 32, l_obs, n_action)
    else:
        net = Net(64, 32, l_obs, n_action)
    net.apply(model.weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    results_reward = []

    for i_iteration in range(NUM_ITER):
        buff = []

        for i_episode in range(NUM_EPISODE):
            net.reset_lstm()

            obs = env.reset()
            obs = np.delete(obs, 1)
            obs = torch.Tensor(obs).unsqueeze(0).unsqueeze(0)

            next_obs = None
            reward = None
            total_reward = 0
            done = False
            traj = Trajectory()

            while not done:
                action = action_decide(net, obs)
                next_obs, reward, done, _ = env.step(action)
                traj.add(obs, action, reward)
                total_reward += reward

                next_obs = np.delete(next_obs, 1)
                obs = next_obs
                obs = torch.Tensor(obs).unsqueeze(0).unsqueeze(0)
                # if _ % 5 == 0 and i_episode == 0:
                #     env.render()
                if done:
                    print('iteration: ', i_iteration, ' episode: ', i_episode, ' reward: ', total_reward)
                    buff.append(traj)
                    results_reward.append(total_reward)
                    writer.add_scalar("Reward/epoch", total_reward, i_iteration * NUM_EPISODE + (i_episode + 1))
        train(net, optimizer, buff)
    env.close()
    writer.flush()
    writer.close()
    return results_reward


def train(net, optimizer, buff):
    optimizer.zero_grad()

    states = [trajectory.get_state() for trajectory in buff]
    actions = [trajectory.get_action() for trajectory in buff]
    returns = [trajectory.get_returns(GAMMA) for trajectory in buff]
    loss_policy = []
    loss_entropy = []

    for i in range(len(buff)):
        net.reset_lstm()

        state = states[i].view(-1, net.l_obs).unsqueeze(0)
        action = actions[i]
        return_ = torch.Tensor(returns[i]).detach()
        return_ = (return_ - return_.mean()) / (return_.std() + eps)

        logits = net(state).view(-1, net.n_action)
        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_act = torch.stack([real_log[real_act] for real_log, real_act in zip(log_prob, action)])

        loss_policy_p = - torch.dot(return_, log_prob_act).view(1) / len(logits)
        loss_policy.append(loss_policy_p)

        loss_entropy_p = - log_prob * prob
        loss_entropy.append(loss_entropy_p)

    loss_policy = torch.cat(loss_policy, dim=0).mean()
    loss_entropy = torch.cat(loss_entropy, dim=0).mean()
    loss = loss_policy + 0.01 * loss_entropy
    loss.backward()
    optimizer.step()


def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        logits = net(obs).view(-1)
        probs = F.softmax(logits, dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


if __name__ == '__main__':
    lr = 0.1
    for _ in range(1):
        results = main()
        plt.figure()
        plt.title("learning rate = %f" % lr)
        plt.plot(results)
        plt.show()
        lr += 0.1
