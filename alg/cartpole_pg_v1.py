"""
Implement of policy gradient in Cartpole environment
Use cumulative reward R as f(s,a) (i.e. REINFORCE algorithm)

version 0.1:
Use a single trajectories to compute policy loss

performance: the learning process works but is unstable with occasional crashing.
"""

import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import model
from model import Net
from model import Trajectory

# Init Hyperparam.
discount_f = 0.9
num_episode = 1000
eps = np.finfo(np.float64).eps.item()
reward_list = []


def main():
    # Init
    # Init env
    env = gym.make('CartPole-v0')

    # Init net model
    # TODO: Find a method to get the shape of obs
    l_obs = 4
    n_action = env.action_space.n
    net = Net(128, 128, l_obs, n_action)
    net.apply(model.weight_init)

    # Init optim
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # episode start
    for i_episode in range(num_episode):
        obs = env.reset()

        next_obs = None
        reward = 0
        total_reward = 0
        done = False
        traj = Trajectory()

        while not done:
            if next_obs is not None:
                obs = next_obs
            obs = torch.tensor(obs).float()
            action = action_decide(net, obs)
            next_obs, reward, done, info = env.step(action)
            traj.add(obs, action, reward)
            total_reward += reward
            if i_episode % 100 == 0:
                env.render()
            if done:
                train(net, optimizer, traj)

        reward_list.append(total_reward)
    env.close()
    return reward_list


def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        probs = F.softmax(net(obs), dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


def train(net, optimizer, trajectory):
    optimizer.zero_grad()

    states = trajectory.get_state()
    actions = trajectory.get_action()
    rewards = trajectory.get_reward()
    R = 0
    returns = []

    # L = -E[log(pi(a | s)) * G]
    logits = net(states)
    log_prob = F.log_softmax(logits, dim=1)
    log_prob_act = torch.stack([log_prob[i][actions[i]] for i in range(len(actions))], dim=0)
    for r in rewards[::-1]:
        R = r + discount_f * R
        returns.insert(0, R)
    returns = torch.Tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    loss_policy = - torch.dot(returns, log_prob_act) / len(logits)

    loss_policy.backward()
    optimizer.step()


if __name__ == '__main__':
    results = main()
    plt.plot(results)
    plt.show()
