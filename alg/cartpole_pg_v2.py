"""
Implement of policy gradient in Cartpole environment
Use cumulative reward R as f(s,a) (i.e. REINFORCE algorithm)

version 0.2:
Use multiple trajectories to compute policy loss with fixed loss function.
Try to utilize Tensorboard to save and observe results.

Performance: Better and more stable than v0.1, which uses just one single trajectory
to compute policy loss in each iteration. Curiously, v0.1 could achieve the upper
reward boundary of the environment faster.
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
from model import Net
from model import Trajectory

# Init Hyperparam.
discount_f = 0.9
num_iter = 5000
num_episode = 1
eps = torch.finfo(torch.float32).eps

seed = 3333
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def main(lr):
    # Init
    # Init env
    env = gym.make('CartPole-v1')

    # Init net model
    # TODO: Find a method to get the shape of obs
    l_obs = env.observation_space.shape[0]
    n_action = env.action_space.n
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dir = 'runs/cartpole_pg_v2_experiment_episode1000/' + date
    writer = SummaryWriter(dir)
    net = Net(128, 128, l_obs, n_action)
    net.apply(model.weight_init)

    # # Inspect the net architecture
    # dummy_input = torch.rand(4)
    # writer.add_graph(net, dummy_input)

    # Init optim
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    reward_list = []

    # Iteration start
    for _ in range(num_iter):
        buff = []

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
                obs = torch.Tensor(obs)
                action = action_decide(net, obs)
                next_obs, reward, done, info = env.step(action)
                traj.add(obs, action, reward)
                total_reward += reward
                # if _ % 5 == 0 and i_episode == 0:
                #     env.render()
                if done:
                    buff.append(traj)
                    reward_list.append(total_reward)
                    writer.add_scalar("Reward/epoch", total_reward, _ * num_episode + (i_episode + 1))

        train(net, optimizer, buff)

    env.close()
    writer.flush()
    writer.close()
    return reward_list


def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        probs = F.softmax(net(obs), dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


def train(net, optimizer, buff):
    optimizer.zero_grad()

    states = [trajectory.get_state() for trajectory in buff]
    actions = [trajectory.get_action() for trajectory in buff]
    rewards = [trajectory.get_reward() for trajectory in buff]
    loss_policy = []
    loss_entropy = []

    for i in range(len(buff)):
        R = 0
        returns = []

        # L = -E[log(pi(a | s)) * G]
        logits = net(states[i])
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=1)

        log_prob_act = torch.stack([log_prob[_][actions[i][_]] for _ in range(len(actions[i]))], dim=0)
        for r in rewards[i][::-1]:
            R = r + discount_f * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        loss_policy_p = - torch.dot(returns, log_prob_act).view(1) / len(logits)
        loss_policy.append(loss_policy_p)

        loss_entropy_p = -log_prob * prob
        loss_entropy.append(loss_entropy_p)

    loss_policy = torch.cat(loss_policy, dim=0).sum() / num_episode
    loss_entropy = torch.cat(loss_entropy, dim=0).sum() / num_episode
    # print(loss_policy.item())
    loss = loss_policy + 0.001 * loss_entropy
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    lr = 0.1
    for _ in range(10):
        results = main(lr)
        plt.figure()
        plt.title("learning rate = %f" % lr)
        plt.plot(results)
        plt.show()
        lr += 0.1
