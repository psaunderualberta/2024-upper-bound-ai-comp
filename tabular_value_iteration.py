from gym_puddle.tabular.value_iteration import *
import gymnasium as gym
from tqdm import tqdm
import os
import json
import pandas as pd
import random

__file_location = os.path.abspath(os.path.dirname(__file__))
__PATH2CONFIGS = os.path.join(__file_location, "gym_puddle", "env_configs")

def get_environment(envno: int):
    config_file = os.path.join(__PATH2CONFIGS, f"pw{envno}.json")
    with open(config_file, "r") as f:
        env = json.load(f)
    return env

def train(nrollouts: int, cutoff: float, granularity: int, envs: list[str], q_table_savepath: str) -> np.ndarray:
    configs = []
    for envno in envs:
        configs.append(get_environment(envno))

    environment = random.choice(configs)
    puddle_top_left = np.array(environment["puddle_top_left"])
    puddle_width = np.array(environment["puddle_width"])

    puddle_agg_func = "sum"

    q_table = vi_create_tables(puddle_top_left, puddle_width, puddle_agg_func, granularity)
    v_table = np.max(q_table, axis=2)
    print(v_table)
    
    pb = tqdm(range(nrollouts))
    for i in pb:
        delta = vi_perform_update(q_table, v_table, puddle_top_left, puddle_width, puddle_agg_func)
        delta = round(delta, 4)
        pb.set_description_str(f"Delta: {delta}")
        if delta < cutoff:
            break

        if i % 50 == 0:
            print(v_table)
    
    vi_save_data(q_table, q_table_savepath)
    df, _ = evaluate(q_table_savepath)
    return df.mean()

def evaluate(filepath):
    q_table = vi_load_data(filepath)
    print(q_table)

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
                    break

                action = q_get_action(q_table, obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                # if done == True:
                #     print("here")

                if done:
                    print(f"total reward in this episode: {total_reward}")
                    total_reward = 0
                    break
            df.loc[seed, f"ep_reward_pw{env}"] = total_reward

            env.close()

    # if episode_rewards == []:
    #     print("no episode finished in this run.")
    # else:
    #     for i, reward in enumerate(episode_rewards):
    #         print(f"episode {i}: reward: {reward}")

    return df.reset_index(names=["seed_ID"]), frames

if __name__ == "__main__":
    train(10000, 1e-5, 10, [1], "q_table.npy")