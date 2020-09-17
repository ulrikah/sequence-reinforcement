import numpy as np

class Metric:
    def __init__(self):
        self.reward_range = None # e.g. (-1, 1) or (-inf, inf)
        self.done_range = None # if the system should operate with a def of done

    def calculate_reward(self, seq_a: np.ndarray, seq_b: np.ndarray):
        raise NotImplementedError

class AbsoluteDifference(Metric):
    def __init__(self):
        super().__init__()
        self.reward_range = [-1.0, 1.0]
   
    def calculate_reward(self, kick_seq: np.ndarray, snare_seq: np.ndarray):
        assert kick_seq.shape == snare_seq.shape
        n_steps = kick_seq.size
        diff = np.abs(kick_seq - snare_seq)
        reward = np.sum(diff / n_steps)
        return np.interp(reward, [0.0, 1.0], self.reward_range)

class NormalizedSum(Metric):
    def __init__(self):
        super().__init__()
        self.reward_range = [-1.0, 1.0]

    def calculate_reward(self, kick_seq: np.ndarray, snare_seq: np.ndarray):
        assert kick_seq.shape == snare_seq.shape
        n_steps = kick_seq.size
        reward = np.sum(snare_seq / n_steps)
        return np.interp(reward, [0.0, 1.0], self.reward_range)
