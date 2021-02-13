"""
implementation of A2C with Pytorch

version: 0.0
Based on a2c_v2
Adding epsilon-greed method to decision making.
"""

import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from matplotlib import pyplot as plt
import torchviz
from cv2 import resize

import model
from model_ac import ConvNet
from model_ac import Trajectory

# Init hyper parameters
env_name = 'Breakout-v0'
num_iter = 200
num_epoch = 25
l1 = 64
l2 = 32
lr = 0.01
gamma = 0.999
reward_list = []

eps = torch.finfo(torch.float32).eps


def main():
    # Init env
    env = gym.make(env_name)
    # l_obs = env.observation_space.shape[0]
    l_obs = 1
    n_action = env.action_space.n

    epsilon = 0.5
    epsilon_end = 0.01
    epsilon_div = 0.025
    epsilon_step = (
            (epsilon - epsilon_end) / epsilon_div)

    # Init net model
    net = ConvNet(l1, l2, l_obs, n_action)
    net.apply(model.weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Iteration start
    for iteration in range(num_iter):
        buff = []
        print("iterations: ", iteration + 1, "/ ", num_iter)

        # Epoch start
        for epoch in range(num_epoch):
            obs = env.reset()

            next_obs = None
            reward = 0
            total_reward = 0
            done = False
            traj = Trajectory()

            while not done:
                if next_obs is None:
                    obs = process_frame(obs)
                    obs = torch.Tensor(obs).unsqueeze(0)
                action = action_decide(net, obs, n_action, epsilon)
                next_obs, reward, done, info = env.step(action)
                next_obs = process_frame(next_obs)
                next_obs = torch.Tensor(next_obs).unsqueeze(0)
                traj.add(obs, action, reward, next_obs)
                total_reward += reward
                obs = next_obs
                if iteration % 10 == 0 and epoch == 1:
                    env.render()
                if done:
                    buff.append(traj)
                    reward_list.append(total_reward)
                    print('reward: ', total_reward)
        A2C(net, optimizer, buff)
        if epsilon > epsilon_end:
            epsilon -= epsilon_step
        else:
            epsilon = epsilon_end
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
        time_step = len(states[i])
        R = 0
        A = []
        V = []
        returns = []
        reward = rewards[i]
        for r in reward[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        log_prob_act = []

        for j in range(time_step):
            # Compute policy loss
            # 1. Get the V value of timestep from critic
            logits, V_s = net(states[i][j])
            logits = logits.view(-1)
            V_s = V_s.view(-1)
            V.append(V_s)

            # Compute adavantage value
            Q_s_a = returns[j]
            A_s_a = Q_s_a - V_s
            A.append(A_s_a)

            # 3. Use V and Q to compute policy loss
            prob = F.softmax(logits, dim=0)
            log_prob = F.log_softmax(logits, dim=0)

            loss_entropy_p = - torch.dot(log_prob, prob)
            loss_entropy.append(loss_entropy_p)

            log_prob_act_p = log_prob[actions[i][j]]
            log_prob_act.append(log_prob_act_p)


        A = torch.Tensor(A)
        A = (A - A.mean()) / (A.std() + eps)
        log_prob_act = torch.stack(log_prob_act, dim=-1)

        loss_policy_p = - torch.dot(A, log_prob_act).view(1) / time_step
        loss_policy.append(loss_policy_p.mean())

        # Compute loss of critic net
        V = torch.Tensor(V)
        loss_critic_p = (returns - V).pow(2).mean()
        loss_critic.append(loss_critic_p)

    loss_policy = torch.stack(loss_policy).mean()
    loss_critic = torch.stack(loss_critic).mean()
    loss_entropy = torch.stack(loss_entropy).mean()
    loss = loss_policy + 0.5 * loss_critic + 0.001 * loss_entropy

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


def action_decide(net, obs, n_action, epsilon=0.0):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        logits, _ = net(obs)
        probs = F.softmax(logits, dim=0)
        probs_hat = (1 - epsilon) * probs + epsilon / n_action
        # print("logits = ", logits, "probs = ", probs)
        m = Categorical(probs_hat)
        action = m.sample()
        return action.item()


def process_frame(frame, crop=34):
    frame = frame[crop:crop + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


if __name__ == '__main__':
    results = main()
    plt.title("learning rate = %f" % lr)
    plt.ylim([-200, 150])
    plt.plot(results)
    plt.show()
