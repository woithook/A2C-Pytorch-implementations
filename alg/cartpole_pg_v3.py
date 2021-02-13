"""
Implement of policy gradient in Cartpole environment
Use cumulative reward R as f(s,a) (i.e. REINFORCE algorithm)

version 0.3: Decouple env and alg.
Use ray.tune to tune hyper parameters
"""

import gym
import ray
from ray import tune
import numpy as np
from numpy import *
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import model
from model import Net
from model import Trajectory

# Init Hyperparam.
num_iter = 100
num_episode = 50
eps = np.finfo(np.float64).eps.item()
reward_list = []
env_name = 'Acrobot-v1'


def train(net, optimizer, discount_f):
    env = gym.make(env_name)
    # Iteration start
    for _ in range(num_iter):
        buff = []

        # episode start
        for i_episode in range(num_episode):
            obs = env.reset()

            next_obs = None
            reward = None
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
                # if _ % 5 == 0 and i_episode == 0:
                #     env.render()
                if done:
                    buff.append(traj)
                    reward_list.append(total_reward)

        pg(net, optimizer, buff, discount_f)

    env.close()
    return reward_list


def test(net):
    env = gym.make(env_name)
    performance = []

    for _ in range(20):
        obs = env.reset()

        next_obs = None
        reward = 0
        total_reward = 0
        done = False
        while not done:
            if next_obs is not None:
                obs = next_obs
            obs = torch.tensor(obs).float()
            action = action_decide(net, obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            # env.render()
            if done:
                performance.append(total_reward)
    performance = mean(performance)
    tune.report(reward_avg=performance)



def action_decide(net, obs):
    # The sampling method that actually based on action distribution
    with torch.no_grad():
        probs = F.softmax(net(obs), dim=0)
        m = Categorical(probs)
        action = m.sample()
        return action.item()


def pg(net, optimizer, buff, discount_f=0.9):
    optimizer.zero_grad()

    states = [trajectory.get_state() for trajectory in buff]
    actions = [trajectory.get_action() for trajectory in buff]
    rewards = [trajectory.get_reward() for trajectory in buff]
    loss_policy = []

    for i in range(len(buff)):
        R = 0
        returns = []

        # L = -E[log(pi(a | s)) * G]
        logits = net(states[i])
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_act = torch.stack([log_prob[_][actions[i][_]] for _ in range(len(actions[i]))], dim=0)
        for r in rewards[i][::-1]:
            R = r + discount_f * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        loss_policy_p = - torch.dot(returns, log_prob_act).view(1) / len(logits)
        loss_policy.append(loss_policy_p)

    loss_policy = torch.cat(loss_policy, dim=0).sum() / num_episode
    # print(loss_policy.item())
    loss_policy.backward()
    optimizer.step()


def tunning_func(config):
    # Hyperparam selected for tunning: discount_f, num_episode, num_fc
    discount_f = config['discount_f']
    num_fc = config['num_fc']
    lr = config['lr']

    env = gym.make(env_name)
    l_obs = 6
    n_action = env.action_space.n
    env.close()

    net = Net(num_fc, num_fc, l_obs, n_action)
    net.apply(model.weight_init)

    # Init optim
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train(net, optimizer, discount_f)
    test(net)


if __name__ == '__main__':

    ray.init(num_cpus=4, include_dashboard=False)
    search_space = {
        'discount_f': tune.grid_search([0.7, 0.8, 0.9]),
        'lr': tune.loguniform(1e-3, 1e-1),
        'num_fc': tune.grid_search([32, 64, 128, 256])
    }
    result = tune.run(tunning_func, config=search_space, num_samples=3)
    df = result.results_df
    best_trial = result.get_best_trial("reward_avg", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
