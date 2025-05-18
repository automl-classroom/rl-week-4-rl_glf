import os
import numpy as np
import torch
from rl_exercises.week_4.dqn import DQNAgent, set_seed
import gymnasium as gym

os.makedirs("results", exist_ok=True)

seeds = [0, 1, 2, 3, 4]
num_frames = 10000
eval_interval = 100
env_name = "CartPole-v1"

for seed in seeds:
    print(f"Running DQN with seed={seed}")
    env = gym.make(env_name)
    set_seed(env, seed)
    agent = DQNAgent(env, seed=seed)
    
    episode_rewards = []
    state, _ = env.reset(seed=seed)
    ep_reward = 0.0

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.buffer.add(state, action, reward, next_state, done or truncated, {})
        state = next_state
        ep_reward += reward

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

        if done or truncated:
            state, _ = env.reset()
            episode_rewards.append(ep_reward)
            ep_reward = 0.0

    rewards_path = f"results/dqn_seed_{seed}.npy"
    np.save(rewards_path, np.array(episode_rewards))
    print(f"Saved rewards to {rewards_path}")