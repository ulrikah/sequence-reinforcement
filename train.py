def train(env, agent, n_episodes, max_steps, batch_size, log=True):
    log_interval = n_episodes // 10
    
    episode_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.get_action(state)
            obs, reward, done, _ = env.step(action)
            # print("\t", "step", step, ":", reward)
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
                print()

    return episode_rewards
