from gym_puddle.tabular.value_iteration import *
import gymnasium as gym
from tqdm import tqdm
import os
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import wandb

__file_location = os.path.abspath(os.path.dirname(__file__))
__PATH2CONFIGS = os.path.join(__file_location, "gym_puddle", "env_configs")

def get_environment(envno: int):
    config_file = os.path.join(__PATH2CONFIGS, f"pw{envno}.json")
    with open(config_file, "r") as f:
        env = json.load(f)
    return env

def train(nrollouts: int, cutoff: float, granularity: int, envs: list[str], q_table_savepath: str) -> np.ndarray:
    # Initialize wandb
    wandb.init(project="upper-bound-2024-comp", entity="psaunder")
    wandb.config.nrollouts = nrollouts
    wandb.config.cutoff = cutoff
    wandb.config.granularity = granularity
    wandb.config.envs = envs

    configs = []
    for envno in envs:
        configs.append(get_environment(envno))

    environment = random.choice(configs)
    puddle_top_left = np.empty((0,0), dtype=np.float64)
    puddle_width = np.empty((0,0), dtype=np.float64)

    puddle_agg_func = "sum"

    q_table, v_table, r_table = vi_create_tables(puddle_top_left, puddle_width, puddle_agg_func, granularity)
    
    pb = tqdm(range(nrollouts))
    for i in pb:
        previous_r_table = r_table.copy()
        previous_v_table = v_table.copy()
        previous_q_table = q_table.copy()
        delta = vi_perform_update(q_table, v_table, r_table, 1.0, puddle_top_left, puddle_width, puddle_agg_func)

        # Log the delta and the change in the r_table, v_table, and q_table
        wandb.log({
            "delta": delta,
            "r_table_change_mu": np.abs(r_table[:, :, 0] - previous_r_table[:, :, 0]).mean(),
            "r_table_change_std": np.abs(r_table[:, :, 0] - previous_r_table[:, :, 0]).std(),
            "r_table_change_max": np.abs(r_table[:, :, 0] - previous_r_table[:, :, 0]).max(),
            "v_table_change_mu": np.abs(v_table - previous_v_table).mean(),
            "v_table_change_std": np.abs(v_table - previous_v_table).std(),
            "v_table_change_max": np.abs(v_table - previous_v_table).max(),
            "q_table_change_mu": np.abs(q_table - previous_q_table).mean(),
            "q_table_change_std": np.abs(q_table - previous_q_table).std(),
            "q_table_change_max": np.abs(q_table - previous_q_table).max()
        })

        delta = round(delta, 3)
        pb.set_description_str(f"Delta: {delta}")
        if delta < cutoff:
            break

        if i % 500 == 0:
            # Upload current q_table, v_table, and r_table to wandb as images and arrays
            # Also, include colorbar
            fig = plt.imshow(np.rot90(np.argmax(q_table, axis=2)), cmap="hot", interpolation="nearest")
            plt.colorbar()
            wandb.log({"action_table": wandb.Image(fig)})
            plt.clf()
            plt.close()

            fig = plt.imshow(np.rot90(v_table), cmap="hot", interpolation="nearest")
            plt.colorbar()
            wandb.log({"value_table": wandb.Image(fig)})
            plt.clf()
            plt.close()

            # v_table changes
            fig = plt.imshow(np.rot90(v_table - previous_v_table), cmap="hot", interpolation="nearest")
            plt.colorbar()
            wandb.log({"value_table_change": wandb.Image(fig)})
            plt.clf()
            plt.close()
    
    vi_save_data(q_table, q_table_savepath)

    # Upload qtable file
    df, _ = evaluate(q_table, save=True)
    print(df.mean())
    wandb.save(q_table_savepath)

    return df.mean()

def evaluate(q_table: np.ndarray, save=False):

    # Create an empty list to store the frames
    frames = []

    df = pd.DataFrame(
        index=range(1, 101),
        columns=["ep_reward_pw1", "ep_reward_pw2", "ep_reward_pw3","ep_reward_pw4", "ep_reward_pw5"
    ])

    for envno in range(1, 6):
        setup = get_environment(envno)
        setup["puddle_agg_func"] = "min"
        setup["start"] = []  # Empty start position to randomize it
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

                action = vi_get_action(q_table, np.array(obs))
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward

                if trunc:
                    print("ERROR: Truncation!")
                    total_reward = 10_000
                    break

                if done:
                    print(f"total reward in this episode: {total_reward}")
                    break

            df.loc[seed, f"ep_reward_pw{envno}"] = total_reward

            env.close()

    # Save to unique file
    if save:
        df.to_csv(os.path.join("dp", "eval.csv"))

    return df.reset_index(names=["seed_ID"]), frames

if __name__ == "__main__":
    train(200000, 1e-5, 50, ["-all"], os.path.join("dp", "q_table.npy"))
