"""
This is the evaluation script for produced reinforcement learning models. 
It takes in a path to a rlgym model, and evaluates computes the total reward across
the 5 environments in gym_puddle/env_configs/pw*.json using seeds 1-100.

It uses the PuddleEnv environment

The output should be a .csv file containing a header in the following format:

seed_ID, ep_reward_pw1, ep_reward_pw2, ep_reward_pw3,ep_reward_pw4, ep_reward_pw5
1, -25, -100, -450,-40, -67
2, -26, -98, -390, -32, -70
3, -35, -90, -480, -89, -45

where:

seed_ID: The seed number for one episode in the test phase. Only evaluate one episode for
each seed and the seed values range from 1 to 100.

ep_reward_pw1, ep_reward_pw2, ep_reward_pw3,ep_reward_pw4, ep_reward_pw5: Episodic rewards (total reward in an episode) obtained
by the RL agent in different configurations of the environment, starting from the first configuration (pw1) till the last (pw5).
These configurations represent variations in the environment's characteristics such as the puddle positions.
"""

import pandas as pd
import os
import json
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
import gym_puddle
import argparse
import sys
import random


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the PuddleWorld environment"
    )
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate")
    args = parser.parse_args()

    model_path = args.model_path

    # Load the model
    model = PPO.load(model_path)

    # Load the environments
    envs = []
    for env_version in range(1, 6):
        json_file = os.path.join(
            os.path.dirname(__file__),
            "gym_puddle",
            "env_configs",
            f"pw{env_version}.json",
        )

        with open(json_file) as f:
            env_setup = json.load(f)
        
        env_setup["puddle_agg_func"] = "min"

        env = gym.make("PuddleWorld-v0", **env_setup)
        envs.append(env)

    # Evaluate the model
    rewards = []
    for seed in range(1, 101):
        rewards.append([seed] + [0] * len(envs))
        for i, env in enumerate(envs):
            total_reward = 0
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward

            rewards[-1][i + 1] = total_reward

    # Save the results
    df = pd.DataFrame(
        rewards,
        columns=[
            "seed_ID",
            "ep_reward_pw1",
            "ep_reward_pw2",
            "ep_reward_pw3",
            "ep_reward_pw4",
            "ep_reward_pw5",
        ],
    )
    df.to_csv("results.csv", index=False)

    print("Results saved to results.csv")


if __name__ == "__main__":
    main()
