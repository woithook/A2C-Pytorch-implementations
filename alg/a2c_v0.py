"""
implementation of A2C with Pytorch

version: 0.0
"""

import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from matplotlib import pyplot as plt

import model
from model_ac import Net
from model_ac import Trajectory

# Init hyper parameters
env_name = 'CartPole-v0'
num_iter = 50
num_epoch = 64
num_fc_a = 64
num_fc_c = 64
gamma = 0.9
eps = np.finfo(np.float64).eps.item()
reward_list = []


def main():
    # Init env
    env = gym.make(env_name)

    l_obs = 4
    n_action = env.action_space.n

    # Init net model
    actor = Net(num_fc_a, num_fc_a, l_obs, n_action)
    critic = Net(num_fc_c, num_fc_c, l_obs, 1)
    actor.apply(model.weight_init)
    critic.apply(model.weight_init)

    optimizer_a = torch.optim.Adam(actor.parameters(), lr=0.01)
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=0.01)

    # Iteration start
    for iteration in range(num_iter):
        buff = []

        # Epoch start
        for epoch in range(num_epoch):
            obs = env.reset()

            next_obs = None
            reward = 0
            total_reward = 0
            done = False
            traj = Trajectory()

            while not done:
                if next_obs is not None:
                    obs = next_obs
                obs = torch.Tensor(obs).float()
                action = action_decide(actor, obs)
                next_obs, reward, done, info = env.step(action)
                next_obs = torch.tensor(next_obs).float()
                traj.add(obs, action, reward, next_obs)
                total_reward += reward
                # if epoch == 1:
                #     env.render()
                if done:
                    buff.append(traj)
                    reward_list.append(total_reward)
        A2C(actor, critic, optimizer_a, optimizer_c, buff)
    env.close()
    return reward_list


def A2C(actor, critic, optimizer_a, optimizer_c, buff):
    optimizer_a.zero_grad()
    optimizer_c.zero_grad()

    states = [trajectory.get_state() for trajectory in buff]
    next_states = [trajectory.get_next_state() for trajectory in buff]
    actions = [trajectory.get_action() for trajectory in buff]
    rewards = [trajectory.get_reward() for trajectory in buff]
    loss_policy = []
    loss_critic = []

    for i in range(len(buff)):

        # Compute policy loss
        # 1. Get the V value of timestep from critic
        with torch.no_grad():
            V_s = critic(states[i])
            # 2. Compute the Q value of timestep t
            V_s_ = critic(next_states[i])
        reward = torch.Tensor(rewards[i])
        # Compute adavantage value
        A_s_a = reward + (gamma * V_s_ - V_s).view(-1)  # TODO: 优势函数的正则化
        A_s_a = (A_s_a - A_s_a.mean()) / (A_s_a.std() + eps)

        # 3. Use V and Q to compute policy loss
        logits = actor(states[i])
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_act = torch.stack([log_prob[_][actions[i][_]] for _ in range(len(actions[i]))], dim=0)
        loss_policy_p = - torch.dot(A_s_a, log_prob_act).view(1) / len(logits)
        loss_policy.append(loss_policy_p)

        # Compute loss of critic net
        V_s = critic(states[i])
        V_s_ = critic(next_states[i])
        loss_critic_p = reward + (V_s_ -V_s).view(-1)
        loss_critic_p = loss_critic_p ** 2
        loss_critic.append(loss_critic_p)

    loss_policy = torch.cat(loss_policy, dim=0).sum() / num_epoch
    loss_policy.backward()
    optimizer_a.step()

    loss_critic = torch.cat(loss_critic, dim=0).sum() / num_epoch
    loss_critic.backward()
    optimizer_c.step()


def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        probs = F.softmax(net(obs), dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


if __name__ == '__main__':
    results = main()
    plt.plot(results)
    plt.show()
