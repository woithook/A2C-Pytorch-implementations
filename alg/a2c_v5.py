"""
implementation of A2C with Pytorch

version: 0.5
Based on v0.3
Use one-trajectory training policy
"""

import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from matplotlib import pyplot as plt
import torchviz

import model
from model_ac import Net
from model import Trajectory


# Init hyper parameters
env_name = 'CartPole-v1'
num_iter = 10000
num_fc_a = 64
num_fc_c = 32
lr = 1e-2
gamma = 0.99
eps = torch.finfo(torch.float32).eps
reward_list = []


def main():
    # Init env
    env = gym.make(env_name)

    l_obs = env.observation_space.shape[0]
    n_action = env.action_space.n

    # Init net model
    net = Net(num_fc_a, num_fc_a, l_obs, n_action)
    net.apply(model.weight_init)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Iteration start
    for iteration in range(num_iter):
        obs = env.reset()

        next_obs = None
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
            traj.add(obs, action, reward)
            total_reward += reward
            # if iteration == 90:
            #     env.render()
            if done:
                reward_list.append(total_reward)
        A2C(net, optimizer, traj)
    env.close()
    return reward_list


def A2C(net, optimizer, trajectory):
    optimizer.zero_grad()

    states = trajectory.get_state()
    actions = trajectory.get_action()
    returns = trajectory.get_returns()

    logits, V_s = net(states)
    logits = logits.view(-1, net.n_action)
    V_s = V_s.view(-1)
    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
    log_prob_act = log_prob.gather(1, actions).view(-1)

    Q_s_a = torch.Tensor(returns)
    Q_s_a = (Q_s_a - Q_s_a.mean()) / (Q_s_a.std() + eps)
    A_s_a = Q_s_a - V_s
    A_s_a = (A_s_a - A_s_a.mean()) / (A_s_a.std() + eps)

    loss_policy = - (A_s_a.detach() * log_prob_act).mean()
    loss_critic = A_s_a.pow(2.).mean()
    loss_entropy = (log_prob * prob).mean()

    loss = loss_policy + .5 * loss_critic + .01 * loss_entropy
    loss.backward()
    # total_norm = 0
    # for p in net.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)
    # print('gradient norm: ', total_norm)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.6)

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
        # print("logits = ", logits, "probs = ", probs)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


if __name__ == '__main__':
    results = main()
    plt.title("learning rate = %f" % lr)
    plt.plot(results)
    plt.show()
