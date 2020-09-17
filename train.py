from time import time, asctime
from collections import Counter
import torch
import numpy as np
from matplotlib import pyplot as plt

def train(env, agent, n_episodes, max_steps, batch_size, 
    eps_decay = 1000,
    render=True, 
    log=True,
    log_interval=100, 
    save_model_to=None, 
    load_model_from=None
):  
    ID = int(time())

    if load_model_from is not None:
        checkpoint = torch.load(load_model_from)
        print(f"💾 Loading model from {load_model_from}")
        print("The loaded model was saved", checkpoint['saved_at'])

        start_from = checkpoint['episode']
        episode_rewards = checkpoint['episode_rewards']
        epsilons = checkpoint['epsilons']
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    else:
        episode_rewards = []
        epsilons = []
        start_from = 0
    
    assert n_episodes > start_from, f"The loaded model has already been trained for more than {n_episodes}"
    
    if log:
        info_interval = (n_episodes - start_from) // log_interval

    actions = []
    for episode in range(start_from, n_episodes + 1):
        state = env.reset()
        episode_reward = []

        eps = epsilon_threshold(episode, eps_decay=eps_decay)
        epsilons.append(eps)
        should_explore = np.random.random_sample() < eps

        for step in range(max_steps):
            if should_explore:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)

            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward.append(reward)

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(np.mean(episode_reward))
                break

            state = next_state

        if log and episode % info_interval == 0:
            print("Mean reward for episode", episode, ":", np.mean(episode_reward))
            if render:
                env.render()
            if save_model_to is not None and episode > start_from:
                path = f"{save_model_to}checkpoint_{episode}_{ID}.cpt"
                save_model(path, 
                {
                    'episode': episode,
                    'episode_rewards': episode_rewards,
                    'epsilons': epsilons,
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'saved_at': asctime()
                })
    if log:
        filename = f"plots/fig_{episode}_{ID}"

        # performance
        plt.subplot(1, 2, 1)
        plt.plot([np.mean(rewards) for rewards in episode_rewards], label="Mean rewards")
        plt.plot(epsilons, label="Epsilon")
        plt.legend()

        # action distribution
        plt.subplot(1, 2, 2)
        labels, values = zip(*sorted(Counter(actions).items()))
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        
        plt.legend()
        plt.savefig(filename)
        print(f"Saved {filename}.png to disk")
        plt.tight_layout()
        plt.show()
    return episode_rewards

def epsilon_threshold(episode, eps_start = 0.9, eps_end = 0.05, eps_decay = 200):
    return eps_end + (eps_start - eps_end) * np.exp(-1. * episode / eps_decay)

def save_model(path, blob):
    print("💾 Saving model to", path)
    torch.save(blob, path)
