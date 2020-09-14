import numpy as np
from matplotlib import pyplot as plt
from time import time

def train(env, agent, n_episodes, max_steps, batch_size, log=True):
    log_interval = n_episodes // 10

    episode_rewards = []
    epsilons = []
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        eps = epsilon_threshold(episode, n_episodes)
        epsilons.append(eps)
        should_explore = np.random.random_sample() > eps

        for step in range(max_steps):
            # exploration
            if should_explore:
                action = env.action_space.sample()
            # exploitation
            else:
                action = agent.get_action(state)

            obs, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, obs, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward / (step + 1))
                break

            state = obs

        if log:
            if episode % log_interval == 0:
                print("Episode", episode, ":", reward)

    if log:
        filename = f"fig_{int(time())}"
        plt.plot(episode_rewards, label="Rewards over time")
        plt.plot(epsilons, label="Epsilon over time")
        plt.legend()
        plt.savefig(f"plots/{filename}")
        # plt.show()
    return episode_rewards

def epsilon_threshold(episode, n_episodes, eps_start = 0.9, eps_end = 0.05, eps_decay = 3):
    return eps_end + (eps_start - eps_end) * np.exp(-1. * (episode / n_episodes) * eps_decay)
