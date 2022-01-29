import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ReplayBuffer:
    def __init__(self, mem_len=100):
        self.episodes = deque(maxlen=mem_len)

    def __len__(self):
        return len(self.episodes)

    def append(self, episode):
        self.episodes.append(episode)

    def sample(self, batch_size):
        return random.sample(self.episodes, batch_size)

class Phi(nn.Module):
    def __init__(self, env_dim, hidden_dim, state_dim):
        super(Phi, self).__init__()
        self.fc1 = nn.Linear(env_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.relu(self.fc3(x))[:, -1] 

class Q(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v0')
    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

    env.close()