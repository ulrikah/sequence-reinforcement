import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from replay_buffer import ReplayBuffer
from dqn import DQN

class Agent:

    def __init__(self, env, in_dim, out_dim, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.model = DQN(in_dim, out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_vals = self.model.forward(state).detach().squeeze()
        distribution = Categorical(q_vals)
        return distribution.sample().item()

    def compute_loss(self, batch):
        states, actions, rewards, next_states, _ = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        curr_q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model.forward(next_states)
        max_next_q = torch.max(next_q, 1)[0]
        expected_q = rewards.squeeze(1) + self.gamma * max_next_q

        return self.loss(curr_q, expected_q)

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()