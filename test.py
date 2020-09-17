from time import time, asctime
from collections import Counter
import torch
import numpy as np
from matplotlib import pyplot as plt

def test(env, agent, checkpoint_path=None):
    assert checkpoint_path is not None, "No checkpoint specified"
    checkpoint = torch.load(checkpoint_path)
    print(f"💾 Loading model from {checkpoint_path}")
    print("The loaded model was saved", checkpoint['saved_at'])

    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    state = env.reset()
    for i in range(10):
        # print(state, "[state]", i)
        action = agent.get_action(state)
        next_state, reward, _, _ = env.step(action)
        # print(next_state, "[next_state]", i)
        print("Reward :", reward, "with action", action)
        env.render()