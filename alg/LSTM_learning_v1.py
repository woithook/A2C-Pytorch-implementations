"""
Code from:
https://blog.csdn.net/hhy_csdn/article/details/106560875
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym

STATE_DIM = 4-1  # 删去小车速度这一维度，使之成为POMDP
ACTION_DIM = 2  # 动作空间大小
NUM_EPISODE = 5000  # 训练的Episode数量
EPISODE_LEN = 1000  # episode最大长度
A_HIDDEN = 40  # Actor网络的隐层神经元数量
C_HIDDEN = 40  # Critic网络的隐层神经元数量

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


class ActorNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        x = F.log_softmax(x, 2)
        return x, hidden


class ValueNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ValueNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


def roll_out(actor_network, value_network, env, episode_len, init_state):
    """
    rollout at most 1000 frames.
    """
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0

    # initialize state and hidden state
    state = init_state
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
    c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
    c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

    # start epoch
    for j in range(episode_len):
        states.append(state)
        log_softmax_action, (a_hx, a_cx) = actor_network(Variable(torch.Tensor([state]).unsqueeze(0)), (a_hx, a_cx))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().data.numpy()[0][0])

        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]

        next_state, reward, done, _ = env.step(action)
        next_state = np.delete(next_state, 1)
        # fix_reward = -10 if done else 1

        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state  # final_state和state是一回事
        state = next_state
        if done:
            is_done = True
            state = env.reset()
            state = np.delete(state, 1)
            a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
            a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
            c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
            c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

            print(j + 1)
            break
    if not is_done:
        c_out, (c_hx, c_cx) = value_network(Variable(torch.Tensor([final_state])), (c_hx, c_cx))
        final_r = c_out.cpu().data.numpy()  # 如果episode正常结束，final_r=0表示终态cart失去控制得0分
    return states, actions, rewards, final_r, state


def discount_reward(r, gamma, final_r):
    """

    :param r: list
    :param final_r: scalar
    :return: returns
    """
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    env = gym.make('CartPole-v1')
    env.seed(seed)

    init_state = env.reset()
    init_state = np.delete(init_state, 1)  # 删除cart velocity维度

    actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=0.001)

    value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=0.005)

    steps = []
    task_episodes = []
    test_results = []

    for episode in range(NUM_EPISODE):
        # 完成一轮rollout
        states, actions, rewards, final_r, current_state = roll_out(
            actor_network, value_network, env, EPISODE_LEN, init_state)

        # 结束rollout后的初态
        init_state = current_state
        actions_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM)).unsqueeze(0)
        states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM)).unsqueeze(0)
        s = states_var.shape

        # 训练动作网络
        # 清空隐藏状态
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

        actor_network_optim.zero_grad()
        # print(states_var.unsqueeze(0).size())
        log_softmax_actions, (a_hx, a_cx) = actor_network(states_var, (a_hx, a_cx))
        vs, (c_hx, c_cx) = value_network(states_var, (c_hx, c_cx))
        vs.detach()

        qs = Variable(torch.Tensor(discount_reward(rewards, 0.99, final_r)))
        qs = qs.view(1, -1, 1)
        advantages = qs - vs
        # print('adv', advantages.shape)
        # log_softmax_actions * actions_var是利用独热编码特性取出对应action的对数概率
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions * actions_var, 1)
                                          * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # 训练价值网络
        value_network_optim.zero_grad()
        target_values = qs
        a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
        values, (c_hx, c_cx) = value_network(states_var, (c_hx, c_cx))

        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
        value_network_optim.step()

        # Testing
        if (episode + 1) % 50 == 0:
            result = 0
            test_task = gym.make("CartPole-v1")
            for test_epi in range(10):  # 测试10个episode
                state = test_task.reset()
                state = np.delete(state, 1)

                a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
                a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
                c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)
                c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0)

                for test_step in range(500):  # 每个episode最长500frame

                    log_softmax_actions, (a_hx, a_cx) = actor_network(Variable(torch.Tensor([state]).view(1, 1, 3)),
                                                                      (a_hx, a_cx))
                    softmax_action = torch.exp(log_softmax_actions)

                    # print(softmax_action.data)
                    action = np.argmax(softmax_action.data.numpy()[0])
                    next_state, reward, done, _ = test_task.step(action)
                    next_state = np.delete(next_state, 1)

                    result += reward
                    state = next_state
                    if done:
                        break
            print("episode:", episode + 1, "test result:", result / 10.0)
            steps.append(episode + 1)
            test_results.append(result / 10)
    plt.plot(steps, test_results)
    plt.savefig('training_score.png')
    plt.show()


if __name__ == '__main__':
    main()
