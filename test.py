from time import time
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

    actions, rewards = [], []
    state = env.reset(snare_seq = np.array([1., .0, .0, .0] * env.n_bars))
    # state = env.reset(snare_seq = np.array([.0, 1., 1., 1.] * env.n_bars))
    for i in range(10):
        action = agent.get_action(state)
        next_state, reward, _, _ = env.step(action)
        print("Reward", i, ":", reward, "with action", action)
        env.render()
        state = next_state

        actions.append(action)
        rewards.append(reward)

    # performance
    plt.subplot(1, 2, 1)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(rewards, label="Rewards in range (-1, 1)")

    # action distribution
    plt.subplot(1, 2, 2)
    labels, values = zip(*sorted(Counter(actions).items()))
    indexes = np.arange(len(labels))
    width = .7
    plt.xlabel("Action number")
    plt.ylabel("Frequency")
    plt.bar(indexes, values, width, label="Actions")
    plt.xticks(indexes + width * 0.5, labels)

    plt.tight_layout()

    # save to disk
    filename = f"plots/fig_test_{int(time())}"
    plt.savefig(filename)
    print(f"Saved {filename}.png to disk")
    
    plt.show()