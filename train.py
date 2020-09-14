import numpy as np
from matplotlib import pyplot as plt
from time import time

def train(env, agent, n_episodes, max_steps, batch_size, render=True, log=True):
    if log:
        log_interval = n_episodes // 10

    episode_rewards = []
    epsilons = []
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = []

        eps = epsilon_threshold(episode, n_episodes)
        epsilons.append(eps)
        should_explore = np.random.random_sample() < eps

        for step in range(max_steps):
            # exploration
            if should_explore:
                action = env.action_space.sample()
            # exploitation
            else:
                action = agent.get_action(state)

            obs, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, obs, done)
            episode_reward.append(reward)

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                break

            state = obs

        if episode % log_interval == 0:
            if log:
                print("Episode", episode, 
                    ":", np.mean(episode_rewards[-1]), 
                    "" if should_explore else "🤖")
            if render:
                env.render()


    if log:
        filename = f"plots/fig_{int(time())}"
        plt.plot([np.mean(rewards) for rewards in episode_rewards], label="Mean rewards")
        plt.plot(epsilons, label="Epsilon")
        plt.legend()
        plt.savefig(filename)
        print(f"Saved {filename}.png to disk")
        plt.show()
    return episode_rewards

def epsilon_threshold(episode, n_episodes, eps_start = 0.9, eps_end = 0.05, eps_decay = 2):
    return eps_end + (eps_start - eps_end) * np.exp(-1. * (episode / n_episodes) * eps_decay)
