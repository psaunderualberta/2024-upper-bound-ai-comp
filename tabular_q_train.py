from gym_puddle.tabular.q_learning import *
import gymnasium as gym
from tqdm import tqdm
import os
import json
import pandas as pd
import random
import wandb
import numpy as np
import matplotlib.pyplot as plt

__file_location = os.path.abspath(os.path.dirname(__file__))
__PATH2CONFIGS = os.path.join(__file_location, "gym_puddle", "env_configs")

def get_environment(envno: int):
    config_file = os.path.join(__PATH2CONFIGS, f"pw{envno}.json")
    with open(config_file, "r") as f:
        env = json.load(f)
    return env

def train(base_nrollouts: int, nupscales: int, init_granularity: int, envs: list[str], q_table_savepath: str) -> np.ndarray:
    # Init wandb
    wandb.init(project="upper-bound-2024-comp", entity="psaunder")
    wandb.config.base_nrollouts = base_nrollouts
    wandb.config.init_granularity = init_granularity
    wandb.config.nupscales = nupscales
    wandb.config.envs = envs

    configs = []
    for envno in envs:
        configs.append(get_environment(envno))

    # Aggregate puddles
    puddle_top_left = []
    puddle_width = []
    puddle_ids = []
    for i, config in enumerate(configs):
        puddle_top_left.extend(config["puddle_top_left"])
        puddle_width.extend(config["puddle_width"])
        puddle_ids.extend([i] * len(config["puddle_top_left"]))
    puddle_top_left = np.array(puddle_top_left)
    puddle_width = np.array(puddle_width)
    puddle_ids = np.array(puddle_ids)

    exploration_factor = 0.05
    lr = 0.01
    gamma = 1.0

    steps = []
    rewards = []
    svpth = f"{q_table_savepath}.npy"
    
    q_table = None
    for i in range(nupscales):

        # Upscale table if not already created
        if q_table is None:
            q_table = q_create_table(init_granularity)
        else:
            # save current table
            svpth = f"{q_table_savepath}-{q_table.shape[0]}x{q_table.shape[1]}.npy"
            q_save_data(q_table, svpth)
            q_table = q_upscale_table(q_table)
        print(f"Upscale {i} - Q Table granularity {q_table.shape[0]} x {q_table.shape[1]}")

        pb = tqdm(range(base_nrollouts), total=base_nrollouts)
        for j in pb:
            if j % 10000 == 0:
                if len(steps) != 0:
                    msteps = round(np.mean(steps), 2)
                    mrewards = round(np.mean(rewards), 2)
                    wandb.log({f"num_steps-{q_table.shape[0]}": msteps})
                    wandb.log({f"cum_rewards-{q_table.shape[0]}": mrewards})

                fig = plt.imshow(np.rot90(np.argmax(q_table, axis=2)), cmap="hot", interpolation="nearest")
                plt.colorbar()
                wandb.log({"action_table": wandb.Image(fig)})
                plt.clf()
                plt.close()

            num_steps, cum_rewards = q_rollout(q_table, exploration_factor, lr, gamma, puddle_top_left, puddle_width, puddle_ids)
            steps.append(num_steps)
            rewards.append(cum_rewards)
            steps = steps[-1000:]
            rewards = rewards[-1000:]

            msteps = round(np.mean(steps), 2)
            mrewards = round(np.mean(rewards), 2)
            pb.set_description_str(f"Steps: {msteps} | Rewards: {mrewards}")

    svpth = f"{q_table_savepath}.npy"
    q_save_data(q_table, svpth)
    df, _ = evaluate(svpth)
    wandb.log({"final_mean_reward": df.mean().mean()})
    return df.mean()

def evaluate(filepath):
    q_table = q_load_data(filepath)

    # Create an empty list to store the frames
    frames = []

    df = pd.DataFrame(
        index=range(1, 101),
        columns=["ep_reward_pw1", "ep_reward_pw2", "ep_reward_pw3","ep_reward_pw4", "ep_reward_pw5"
    ])

    for env in range(1, 6):
        setup = get_environment(env)
        setup["start"] = []
        for seed in range(1, 101):
            env = gym.make("PuddleWorld-v0", **setup)
            obs, _ = env.reset()
            total_reward = 0
            done = False
            num_steps = 0

            while not done: # to avoid infinite loops for the untuned DQN we set a truncation limit, but you should make your agent sophisticated enough to avoid infinite-step episodes
                num_steps +=1

                if num_steps > 10000:
                    print("ERROR: Exceeded 10000 steps!")
                    total_reward = np.inf
                    break

                action = q_get_action(q_table, obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward

                if done:
                    print(f"total reward in this episode: {total_reward}")
                    total_reward = 0
                    break
            df.loc[seed, f"ep_reward_pw{env}"] = total_reward

            env.close()

    return df.reset_index(names=["seed_ID"]), frames

if __name__ == "__main__":
    train(int(1e6), 8, 50, [1, 2, 3, 4, 5], os.path.join("dp", "q_table"))