import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_vals = self.model.forward(state).detach().squeeze()
        distribution = Categorical(q_vals)
        return distribution.sample().item()

    def compute_loss(self, batch):
        states, actions, rewards, next_states, _ = batch
        state_batch = torch.FloatTensor(states)
        action_batch = torch.LongTensor(actions)
        reward_batch = torch.FloatTensor(rewards)
        next_state_batch = torch.FloatTensor(next_states)

        state_action_values = self.model.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_state_values = self.model.forward(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(1)

        return F.smooth_l1_loss(state_action_values, expected_state_action_values)

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()