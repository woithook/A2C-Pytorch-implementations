"""
implementation of A2C with Pytorch

version: 0.5
Based on Pong v0.0
Train agent to learn to play Breakout with fc networks.
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
from model_ac import Net, ConvNet
from model import Trajectory
from utils import atari_env


LEARNING_RATE = 3e-4
GAMMA = 0.99
NUM_ITER = 50000
eps = torch.finfo(torch.float32).eps

env_name = 'Breakout-v0'
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

    net = ConvNet(64, 32, l_obs, n_action).to(DEVICE)
    net.apply(model.weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    results_reward = []

    for i_iteration in range(NUM_ITER):
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
            if i_iteration % 100 == 0:
                env.render()
                time.sleep(1/60)
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

    optimizer.zero_grad()

    states = states.to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(DEVICE)
    return_ = torch.Tensor(returns).to(DEVICE)

    logits = [None for _ in range(len(states))]
    v = [None for _ in range(len(states))]
    for i in range(len(states)):
        logits_p, v_p = net(states[i])
        logits[i] = logits_p.view(-1)
        v[i] = v_p.view(-1)

    logits = torch.stack(logits, dim=0)
    v = torch.stack(v, dim=-1)
    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    log_prob_act = log_prob.gather(1, actions).view(-1)

    q = return_
    q = (q - q.mean()) / (q.std() + eps)
    a = q - v
    a = (a - a.mean()) / (a.std() + eps)

    loss_policy = - (a.detach() * log_prob_act).sum()
    loss_critic = a.pow(2.).sum()
    loss_entropy = (log_prob * prob).sum()

    loss = loss_policy + .5 * loss_critic + .01 * loss_entropy
    loss.backward()
    # total_norm = 0
    # for p in net.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 40)

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
