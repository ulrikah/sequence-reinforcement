import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from replay_buffer import ReplayBuffer
from dqn import DQN

class Agent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        self.model = DQN(env.action_space.n)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).float().unsqueeze(0)
        qvals = self.model.forward(state)
        # quantize to 0 or 1
        action = np.where(qvals.detach().numpy() >= 0.5, 1, 0).squeeze()
        
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # import pdb; pdb.set_trace()

        curr_q = self.model.forward(states).gather(1, actions)
        next_q = self.model.forward(next_states)
        expected_q = rewards + self.gamma * next_q

        loss = self.loss(curr_q, expected_q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()