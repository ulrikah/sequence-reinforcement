import gym
import numpy as np

class SequenceActions:
    # TO DO: inherit from gym.Space
    def __init__(self, n_steps):
        self.n = n_steps
    
    def sample(self):
        return np.random.rand(self.n,)

class MidiEnv(gym.Env):
    def __init__(self, n_bars = 4, beats_per_bar = 4, metric = "diff"):
        '''Initialises a np array of fixed length for kick and snare'''
        self.n_bars = n_bars
        self.beats_per_bar = beats_per_bar
        
        self.kick_seq = np.array([1, 0, 0, 0] * self.n_bars)
        self.snare_seq = np.zeros(self.get_n_steps())
        
        self.action_space = SequenceActions(n_steps = self.get_n_steps())
        self.metric = metric
        
        if self.metric == "diff":
            self.reward_range = (0.0, 1.0)
            self.done_range = (0.5, 1.0)
        
        if self.metric == "snare_sum":
            self.reward_range = (0.0, 1.0)
            self.done_range = (0.2, 0.9)
    
    def get_n_steps(self):
        return self.n_bars * self.beats_per_bar
    
    def get_state(self):
        return self.kick_seq
    
    def __len__(self):
        return self.get_n_steps()

    def step(self, action : np.ndarray):
        self.snare_seq = action
        reward = self.calculate_reward(self.kick_seq, action)
        
        '''
        # episodes that go outside this range should be considered done
        _min, _max = self.done_range
        done = not _min <= reward <= _max
        '''
        done = False

        return self.kick_seq, reward, done, {}
    
    def reset(self):
        self.kick_seq = np.array([1, 0, 0, 0] * self.n_bars)
        # self.kick_seq = np.random.choice([0, 1], size=(self.get_n_steps(),), p=[3./4, 1./4])
        self.snare_seq = np.zeros(self.get_n_steps())
        return self.get_state()

    def close(self):
        return

    def calculate_reward(self, kick_seq : np.ndarray, snare_seq: np.ndarray):
        '''Normalized sum to fit in the reward range'''
        assert kick_seq.shape == snare_seq.shape

        if self.metric == "diff":
            diff = np.abs(kick_seq - snare_seq)
            return np.sum(diff / self.get_n_steps())

        if self.metric == "snare_sum":
            return np.sum(snare_seq / self.get_n_steps())

    def render(self, log=True):
        '''Renders the np array as a sequence of MIDI notes'''
        if log:
            print(self.to_string(np.where(self.snare_seq >= 0.5, 1, 0)), "SNARE", "\t", np.round(self.snare_seq, 2))
            print(self.to_string(self.kick_seq), "KICK", "\t", np.round(self.kick_seq, 2))

        # TO DO: render as MIDI and send on output port

    def to_string(self, seq : np.ndarray):
        x = lambda s : str(s).strip().replace("1", "x").replace("0", "o")
        sequence = list(map(x, seq.tolist()))
        return " ".join(sequence)
