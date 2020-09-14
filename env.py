import gym
import numpy as np

# inspired by RaveForce: https://github.com/chaosprint/RaveForce/blob/master/Python/raveforce.py
# also contains many more action spaces

# TO DO: inherit from gym.Space
class SequenceActions:
    def __init__(self, n_steps):
        self.n = n_steps
    
    def sample(self):
        return np.random.choice([0, 1], size=(self.n,))
    

class MidiEnv(gym.Env):
    def __init__(self, n_bars = 4, beats_per_bar = 4):
        '''Initialises a np array of fixed length for kick and snare'''
        self.n_bars = n_bars
        self.beats_per_bar = beats_per_bar
        
        self.kick_seq = np.array([1, 0, 0, 0] * self.n_bars)
        # self.kick_seq = np.random.choice([0, 1], size=(get_n_steps(),), p=[3./4, 1./4])
        self.snare_seq = np.zeros(self.get_n_steps(), dtype = np.int8)
        
        self.action_space = SequenceActions(n_steps = self.get_n_steps())
        self.reward_range = (0.0, 1.0)
        self.done_threshold = (0.1, 0.9) # episodes that go outside this range should be considered done
        
        # TO DO - how to define the observation space? it's just a np array of length n_steps
        self.observation_space = None

    
    def get_n_steps(self):
        return self.n_bars * self.beats_per_bar
    
    def get_state(self):
        return self.kick_seq
    
    def __len__(self):
        return self.get_n_steps()

    def step(self, action : np.ndarray):
        reward = self.calculate_reward(self.kick_seq, action)
        done = False
        _min, _max = self.done_threshold
        if not (_min <= reward <= _max):
            done = True
        self.snare_seq = action
        return self.kick_seq, reward, done, {}
    
    def reset(self):
        self.kick_seq = np.array([1, 0, 0, 0] * self.n_bars)
        # self.kick_seq = np.random.choice([0, 1], size=(get_n_steps(),), p=[3./4, 1./4])
        self.snare_seq = np.zeros(self.get_n_steps(), dtype = np.int8)
        return self.get_state()
        
    def close(self):
        return
    
    def calculate_reward(self, kick_seq : np.ndarray, snare_seq: np.ndarray):
        assert kick_seq.shape == snare_seq.shape
        diff = np.sum(kick_seq - np.where(snare_seq > 0.5, 1, 0))
        return np.abs(diff/self.get_n_steps()) # normalized sum to fit in the reward range
    
    def render(self):
        '''Renders the np array as a sequence of MIDI notes'''
        print("Snare sequence", self.snare_seq)
        print("Kick sequence", self.kick_seq)
        return