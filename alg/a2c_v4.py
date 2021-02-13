"""
implementation of A2C with Pytorch

version: 0.4
Adding LSTM layer to accelerate training
"""

import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from matplotlib import pyplot as plt
import torchviz

import model
from model_ac import Net_LSTM
from model_ac import Trajectory


# Init hyper parameters
env_name = 'CartPole-v1'
num_iter = 100
num_epoch = 64
num_fc_a = 128
num_fc_c = 128
lr = 0.003
beta1 = 0.9
beta2 = 0.999
gamma = 0.999
eps = 1e-08
reward_list = []


def main():
    # Init env
    env = gym.make(env_name)

    l_obs = env.observation_space.shape[0]
    n_action = env.action_space.n

    # Init net model
    net = Net_LSTM(num_fc_a, num_fc_a, l_obs, n_action)
    net.apply(model.weight_init)

    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

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
                action = action_decide(net, obs)
                next_obs, reward, done, info = env.step(action)
                next_obs = torch.tensor(next_obs).float()
                traj.add(obs, action, reward, next_obs)
                total_reward += reward
                # if iteration == 90:
                #     env.render()
                if done:
                    buff.append(traj)
                    reward_list.append(total_reward)
        A2C(net, optimizer, buff)
    env.close()
    return reward_list


def A2C(net, optimizer, buff):
    optimizer.zero_grad()

    states = [trajectory.get_state() for trajectory in buff]
    next_states = [trajectory.get_next_state() for trajectory in buff]
    actions = [trajectory.get_action() for trajectory in buff]
    rewards = [trajectory.get_reward() for trajectory in buff]
    loss_policy = []
    loss_critic = []
    loss_entropy = []

    for i in range(len(buff)):
        R = 0
        returns = []

        # Compute policy loss
        # 1. Get the V value of timestep from critic
        logits, V_s = net(states[i])
        # 2. Compute the Q value of timestep t
        _, V_s_ = net(next_states[i])
        V_s = V_s.view(-1)
        V_s_ = V_s_.view(-1)

        # Compute adavantage value
        reward = rewards[i]
        for r in reward[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        reward = torch.Tensor(reward)
        returns = torch.Tensor(returns)
        Q_s_a = torch.Tensor(returns)

        A_s_a = Q_s_a - V_s
        # A_s_a = (A_s_a - A_s_a.mean()) / (A_s_a.std() + eps)

        # 3. Use V and Q to compute policy loss
        prob = F.softmax(logits, dim=0)
        log_prob = F.log_softmax(logits, dim=1)
        loss_entropy_p = - log_prob * prob
        loss_entropy_p = loss_entropy_p.mean()
        loss_entropy.append(loss_entropy_p)

        log_prob_act = torch.stack([log_prob[_][actions[i][_]] for _ in range(len(actions[i]))], dim=0)
        loss_policy_p = - torch.dot(A_s_a.detach(), log_prob_act).view(1) / len(logits)
        # loss_policy_p = - torch.dot(A_s_a, log_prob_act).view(1) / len(logits)
        loss_policy_p = loss_policy_p.mean()
        loss_policy.append(loss_policy_p)

        # Compute loss of critic net
        loss_critic_p = (returns - V_s).pow(2).mean()
        # loss_critic_p = A_s_a.pow(2).mean()
        loss_critic.append(loss_critic_p)

    loss_policy = torch.stack(loss_policy).mean()
    loss_critic = torch.stack(loss_critic).mean()
    loss_entropy = torch.stack(loss_entropy).mean()
    loss = loss_policy + 0.5 * loss_critic + 0.001 * loss_entropy
    # loss = loss_policy + loss_critic

    loss.backward()

    check_grad = False
    if check_grad:
        for name, weight in net.named_parameters():
            # print("weight:", weight) # 打印权重，看是否在变化
            if weight.requires_grad:
                # print("name", name, "weight:", weight.grad)  # 打印梯度，看是否丢失
                # 直接打印梯度会出现太多输出，可以选择打印梯度的均值、极值，但如果梯度为None会报错
                print("name", name, "weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("---------------------------------"
              "----------------\n")
    optimizer.step()


def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        logits, _ = net(obs)
        probs = F.softmax(logits, dim=0)
        print("logits = ", logits, "probs = ", probs)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


if __name__ == '__main__':
    results = main()
    plt.title("learning rate = %d" % lr)
    plt.plot(results)
    plt.show()
