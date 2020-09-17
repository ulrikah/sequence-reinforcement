import gym
import numpy as np
from metrics import AbsoluteDifference

class SimpleEnv(gym.Env):
    def __init__(self, n_bars, metric = AbsoluteDifference):
        self.n_bars = n_bars
        self.n_steps = n_bars * 4
        
        self.action_space = gym.spaces.Discrete(self.n_steps + 1)

        self.kick_seq = np.array([1.0, .0, .0, .0] * self.n_bars)
        self.snare_seq = np.random.choice([0, 1], self.n_steps)

        self.metric = metric()
        self.reward_range = self.metric.reward_range

    def step(self, action : int):
        '''Performs one step in the environment where action is an integer of 0 - n_steps'''
        assert self.action_space.contains(action), "Action is outside the action space"

        if action < self.n_steps:
            self.snare_seq[action] = 1 - self.snare_seq[action]

        reward = self.calculate_reward(self.kick_seq, self.snare_seq)
        
        done = False
        return self.get_state(), reward, done, {}

    def get_state(self):
        assert self.kick_seq.shape == self.snare_seq.shape, "The two patterns don't have the same shape"
        return np.concatenate((self.kick_seq, self.snare_seq))

    def reset(self):
        self.kick_seq = np.array([1.0, .0, .0, .0] * self.n_bars)
        self.snare_seq = np.random.choice([0, 1], self.n_steps)
        return self.get_state()

    def close(self):
        return

    def calculate_reward(self, kick_seq, snare_seq):
        return self.metric.calculate_reward(kick_seq, snare_seq)

    def render(self, log=True):
        '''Renders the np array as a sequence of MIDI notes'''
        if log:
            print(self.to_string(self.snare_seq), "SNARE")
            print(self.to_string(self.kick_seq), "KICK")

    def to_string(self, seq : np.ndarray):
        x = lambda s : str(s).strip().replace("1", "x").replace("0", "o")
        sequence = list(map(x, seq.astype(np.int8).tolist()))
        return " ".join(sequence)