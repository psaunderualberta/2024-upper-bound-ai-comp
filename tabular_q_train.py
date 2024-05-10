from gym_puddle.tabular.q_learning import *
import gymnasium as gym
from tqdm import tqdm
import os
import json
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse

__file_location = os.path.abspath(os.path.dirname(__file__))
__PATH2CONFIGS = os.path.join(__file_location, "gym_puddle", "env_configs")


def get_environment(envno: int):
    """
    Get the environment configuration for the given environment number
    """
    config_file = os.path.join(__PATH2CONFIGS, f"pw{envno}.json")
    with open(config_file, "r") as f:
        env = json.load(f)
    return env


def train(config: dict) -> np.ndarray:
    """_summary_

    Args:
        config (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    base_nrollouts = config["nrollouts"]
    nupscales = config["nupscales"]
    init_granularity = config["init_granularity"]
    envs = config["envs"].split(",")
    q_table_savepath = config["q_table_savepath"]

    # Init wandb
    if config["wandb"]:
        wandb.init(project="upper-bound-2024-comp", entity="psaunder")
        wandb.config.update(config)

    env_configs = []
    for envno in envs:
        env_configs.append(get_environment(envno))

    # Aggregate puddles
    puddle_top_left = []
    puddle_width = []
    puddle_ids = []
    for i, cf in enumerate(env_configs):
        puddle_top_left.extend(cf["puddle_top_left"])
        puddle_width.extend(cf["puddle_width"])
        puddle_ids.extend([i] * len(cf["puddle_top_left"]))
    puddle_top_left = np.array(puddle_top_left)
    puddle_width = np.array(puddle_width)
    puddle_ids = np.array(puddle_ids)

    exploration_factor = 0.3
    lr = 0.01
    gamma = 1.0

    steps = []
    rewards = []
    svpth = f"{q_table_savepath}.npy"

    q_table = None
    for i in range(nupscales):
        if q_table is None:
            # Load table if it exists, else create a new one
            if config["q_table_path"] is not None and os.path.isfile(config["q_table_path"]):
                q_table = q_load_data(config["q_table_path"])
            else:
                q_table = q_create_table(init_granularity)
        else:
            # save current table
            svpth = f"{q_table_savepath}-{q_table.shape[0]}x{q_table.shape[1]}.npy"
            q_save_data(q_table, svpth)

            # Upscale table
            q_table = q_upscale_table(q_table)
        print(
            f"Upscale {i} - Q Table granularity {q_table.shape[0]} x {q_table.shape[1]}"
        )

        pb = tqdm(range(base_nrollouts), total=base_nrollouts)
        for j in pb:
            if config["wandb"] and j % 10000 == 0:
                if len(steps) != 0:
                    msteps = round(np.mean(steps), 2)
                    mrewards = round(np.mean(rewards), 2)
                    wandb.log({f"num_steps-{q_table.shape[0]}": msteps})
                    wandb.log({f"cum_rewards-{q_table.shape[0]}": mrewards})

                    cum_rewards = 0
                    start = np.array([0.2, 0.4])
                    for env in envs:
                        setup = get_environment(env)
                        puddle_top_left_tmp = np.array(setup["puddle_top_left"])
                        puddle_width_tmp = np.array(setup["puddle_width"])
                        puddle_ids_tmp = np.array([0] * len(puddle_top_left))

                        for _ in range(100):
                            _, cum_reward = q_rollout(
                                q_table,
                                0.0,
                                lr,
                                gamma,
                                puddle_top_left_tmp,
                                puddle_width_tmp,
                                puddle_ids_tmp,
                                start,
                            )
                            cum_rewards += cum_reward
                    wandb.log(
                        {f"cum_rewards-test-{q_table.shape[0]}": cum_rewards / 500}
                    )

                fig = plt.imshow(
                    np.rot90(np.argmax(q_table, axis=2)),
                    cmap="hot",
                    interpolation="nearest",
                )
                plt.colorbar()
                wandb.log({"action_table": wandb.Image(fig)})
                plt.clf()
                plt.close()

            ## Perform a single rollout
            num_steps, cum_rewards = q_rollout(
                q_table,
                exploration_factor,
                lr,
                gamma,
                puddle_top_left,
                puddle_width,
                puddle_ids,
            )
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

    if config["wandb"]:
        wandb.log({"final_mean_reward": df.mean().mean()})
    return df.mean()


def evaluate(filepath, save=False):
    q_table = q_load_data(filepath)

    # Create an empty list to store the frames
    frames = []

    df = pd.DataFrame(
        index=range(1, 101),
        columns=[
            "ep_reward_pw1",
            "ep_reward_pw2",
            "ep_reward_pw3",
            "ep_reward_pw4",
            "ep_reward_pw5",
        ],
    )

    for envno in range(1, 6):
        setup = get_environment(envno)
        for seed in range(1, 101):
            env = gym.make("PuddleWorld-v0", **setup)
            obs, _ = env.reset(seed=seed)
            total_reward = 0
            done = False
            num_steps = 0

            while (
                not done
            ):  # to avoid infinite loops for the untuned DQN we set a truncation limit, but you should make your agent sophisticated enough to avoid infinite-step episodes
                num_steps += 1

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
            df.loc[seed, f"ep_reward_pw{envno}"] = total_reward

            env.close()

    print(df.mean(), df.mean().mean())
    df = df.reset_index(names=["seed_ID"])

    if save:
        df.to_csv(f"dp/eval.csv")

    return df, frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrollouts", type=int, default=int(1e6), help="Number of rollouts to perform per upscaled table")
    parser.add_argument("--nupscales", type=int, default=8, help="Number of times to upscale the table")
    parser.add_argument("--upscale-factor", type=int, default=2, help="Factor to upscale the table by")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Q learning")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for Q learning")
    parser.add_argument("--exploration-factor", type=float, default=0.3, help="Exploration factor for Q learning")
    parser.add_argument("--init_granularity", type=int, default=50, help="Initial granularity of the Q table")
    parser.add_argument("--envs", type=str, default="1,2,3,4,5", help="Environments to train on")
    parser.add_argument("--q_table_savepath", type=str, default=os.path.join("dp", "q_table"), help="Path to save the Q table")
    parser.add_argument("--wandb", action='store_true', help="Log to wandb")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate the Q table")
    parser.add_argument("--train", action='store_true', help="Train the Q table")
    parser.add_argument("--q_table_path", type=str, default=None, help="Path to the Q table to evaluate")
    args = parser.parse_args()

    if args.train:
        train(vars(args))

    if args.evaluate:
        assert os.path.isfile(args.q_table_path), f"Q Table path {args.q_table_path} does not exist"
        evaluate(args.q_table_path, save=True)
