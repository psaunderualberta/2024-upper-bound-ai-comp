import gymnasium as gym
import gym_puddle

from stable_baselines3 import PPO
from stable_baselines3.dqn import MlpPolicy as DQNPolicy

# import matplotlib.pyplot as plt
# import numpy as np

from util import visualize

import os


def train():
    #train the model, and save the trained model
    env = gym.make("NoPuddleWorldStochastic-v0")
    ppo_model = PPO('MlpPolicy', env, verbose=1)
    ppo_model.learn(total_timesteps=int(1e5))   
    ppo_model.save("ppo_model")

def enjoy():
    env = gym.make("NoPuddleWorldStochastic-v0")
    ppo_model = PPO.load("ppo_model.zip")

    obs, info = env.reset()

    # Create an empty list to store the frames
    frames = []
    episode_rewards = []

    for episode in range(1):
        total_reward = 0
        done = False
        num_steps = 0

        while not done and num_steps <=1000: # to avoid infinite loops for the untuned DQN we set a truncation limit, but you should make your agent sophisticated enough to avoid infinite-step episodes
            num_steps +=1
            action, _states = ppo_model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward

            image = env.render()
            frames.append(image)

            if done:
                print(f"total reward in this episode: {total_reward}")
                episode_rewards.append(total_reward)
                total_reward = 0
                break

    env.close()

    if episode_rewards == []:
        print("no episode finished in this run.")
    else:
        for i, reward in enumerate(episode_rewards):
            print(f"episode {i}: reward: {reward}")

    video_file = os.path.join("videos", "PPO.mp4")
    visualize(frames, video_file)



if __name__ == "__main__":
    # train()
    enjoy()