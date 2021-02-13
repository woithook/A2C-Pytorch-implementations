"""
Implement of policy gradient in Cartpole environment
Use cumulative reward R as f(s,a) (i.e. REINFORCE algorithm)

version 0.0:
Use multiple trajectories to compute policy loss.

Wrong point: The computation of Gt is wrong, actually it should be computed reversely.

Performance: The agent absolutely cannot learning any thing.
"""
import math
import ray
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import model
from model import Net


def main():
    # Init
    # Init env
    env = gym.make('CartPole-v0')

    # Init Hyperparam.
    discount_f = 1
    num_episode = 64
    num_iter = 100
    reward_list = []

    # Init net model
    # TODO: Find a method to get the shape of obs
    l_obs = 4
    n_action = env.action_space.n
    net = Net(128, 128, l_obs, n_action)
    net.apply(model.weight_init)

    # Init optim
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # iteration
    for i in range(num_iter):
        # episode rollout
        traj = model.Trajectory()
        for episode in range(num_episode):
            obs = env.reset()

            step = 0
            next_obs = None
            step_reward = 0
            total_reward = 0
            done = False

            while not done:
                if next_obs is not None:
                    obs = next_obs
                obs = torch.tensor(obs).float()
                action = action_decide(net, obs)
                next_obs, step_reward, done, info = env.step(action)
                total_reward += (discount_f ** step) * step_reward
                traj.add(obs, action, total_reward)

                if done:
                    reward_list.append(total_reward)

                step += 1

        train(net, optimizer, traj)

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
    action = trajectory.get_action()
    reward = trajectory.get_reward()

    # L = -E[log(pi(a | s)) * G]
    logits = net(states)
    log_prob = F.log_softmax(logits, dim=1)
    log_prob_act = torch.stack([log_prob[i][action[i]] for i in range(len(action))], dim=0)
    loss_policy = - torch.dot(reward, log_prob_act)

    loss_policy.backward()
    optimizer.step()


if __name__ == '__main__':

    results = main()
    plt.plot(results)
    plt.show()
