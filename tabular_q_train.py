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
    """
    Train the Q table on the given environments

    Args:
        config (dict): Configuration dictionary with the following keys:

    Returns:
        np.ndarray: Mean reward of the trained Q table
    """
    base_nrollouts = config["nrollouts"]
    nupscales = config["nupscales"]
    init_granularity = config["init_granularity"]
    envs = config["envs"].split(",")
    q_table_savepath = config["q_table_savepath"]

    # Init wandb
    if config["wandb"]:
        wandb.init(project=config["wandb-project"], entity=config["wandb-entity"])
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

    exploration_factor = config["exploration_factor"]
    lr = config["lr"]
    gamma = config["gamma"]

    # Check if a starting position is provided, else None
    start_pos = (
        map(float, config["start_pos"].split(","))
        if config["start_pos"] is not None
        else None
    )
    start_pos = np.array(list(start_pos)) if start_pos is not None else None

    steps = []
    rewards = []
    svpth = f"{q_table_savepath}.npy"

    q_table = None
    for i in range(nupscales):
        if q_table is None:
            # Load table if it exists, else create a new one
            if config["q_table_path"] is not None and os.path.isfile(
                config["q_table_path"]
            ):
                q_table = q_load_data(config["q_table_path"])
            else:
                q_table = q_create_table(init_granularity)
        else:
            # save current table
            svpth = f"{q_table_savepath}-{q_table.shape[0]}x{q_table.shape[1]}.npy"
            q_save_data(q_table, svpth)

            # Upscale table
            q_table = q_upscale_table(q_table, config["upscale_factor"])

        # Train the Q table
        pb = tqdm(range(base_nrollouts), total=base_nrollouts)
        print(
            f"Upscale {i} - Q Table granularity "
            + f"{q_table.shape[0]} x {q_table.shape[1]}"
        )
        for j in pb:

            # Log information to wandb every 10000 steps
            # Wandb doesn't like doing it super frequently
            if config["wandb"] and j % 10000 == 0:
                if len(steps) != 0:
                    # Log average of last 1000 steps and rewards
                    msteps = round(np.mean(steps), 2)
                    mrewards = round(np.mean(rewards), 2)
                    wandb.log({f"num_steps-{q_table.shape[0]}": msteps})
                    wandb.log({f"cum_rewards-{q_table.shape[0]}": mrewards})

                    # Log average of 500 rollouts from the test's start state
                    # No exploration
                    cum_rewards = 0
                    start = np.array([0.2, 0.4])
                    for env in envs:
                        setup = get_environment(env)
                        puddle_top_left_tmp = np.array(setup["puddle_top_left"])
                        puddle_width_tmp = np.array(setup["puddle_width"])
                        puddle_ids_tmp = np.array([0] * len(puddle_top_left))

                        # Not exactly, but approximately equal to the 100 seeds
                        # when evaluating the learned model
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

                # Plot the actions table
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
                start_pos,
            )

            # Store the last 1000 steps and rewards
            steps.append(num_steps)
            rewards.append(cum_rewards)
            steps = steps[-1000:]
            rewards = rewards[-1000:]

            # Set the progress bar description
            msteps = round(np.mean(steps), 2)
            mrewards = round(np.mean(rewards), 2)
            pb.set_description_str(f"Steps: {msteps} | Rewards: {mrewards}")

    # Save the final Q table
    svpth = f"{q_table_savepath}.npy"
    q_save_data(q_table, svpth)
    df = evaluate(svpth)

    # Log the final mean reward to wandb
    if config["wandb"]:
        wandb.log({"final_mean_reward": df.mean().mean()})
    return df.mean().mean()


def evaluate(filepath, save=False):
    """
    Evaluate the Q table on the 5 environments

    Args:
        filepath (str): Path to the Q table
        save (bool): Whether to save the evaluation results
    """
    q_table = q_load_data(filepath)

    # Create a dataframe to store the result
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
            ):  # to avoid infinite loops for the untuned DQN, use a truncation limit
                num_steps += 1

                if num_steps > 10000:
                    # This should never happen with a well-trained model
                    print("ERROR: Exceeded 10000 steps!")
                    total_reward = np.inf
                    break

                action = q_get_action(q_table, obs)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward

                if done:
                    print(f"total reward in this episode: {total_reward}")
                    break
            df.loc[seed, f"ep_reward_pw{envno}"] = total_reward

            env.close()

    print(df.mean(), df.mean().mean())
    df = df.reset_index(names=["seed_ID"])

    # Save the evaluation results
    if save:
        df.to_csv(f"dp/eval.csv", index=False)

    return df


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nrollouts",
        type=int,
        default=int(1e6),
        help="Number of rollouts to perform per upscaled table",
    )
    parser.add_argument(
        "--nupscales", type=int, default=8, help="Number of times to upscale the table"
    )
    parser.add_argument(
        "--upscale-factor", type=int, default=2, help="Factor to upscale the table by"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for Q learning"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Discount factor for Q learning"
    )
    parser.add_argument(
        "--exploration-factor",
        type=float,
        default=0.3,
        help="Exploration factor for Q learning",
    )
    parser.add_argument(
        "--init_granularity",
        type=int,
        default=50,
        help="Initial granularity of the Q table",
    )
    parser.add_argument(
        "--envs", type=str, default="1,2,3,4,5", help="Environments to train on"
    )
    parser.add_argument(
        "--q_table_savepath",
        type=str,
        default=os.path.join("dp", "q_table"),
        help="Path to save the Q table",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="upper-bound-2024-comp")
    parser.add_argument("--wandb-entity", type=str, default="psaunder")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the Q table")
    parser.add_argument("--train", action="store_true", help="Train the Q table")
    parser.add_argument(
        "--q_table_path", type=str, default=None, help="Path to the Q table to evaluate"
    )
    parser.add_argument(
        "--start-pos", type=str, default=None, help="Starting position for training"
    )
    args = parser.parse_args()

    if args.train:
        train(vars(args))

    if args.evaluate:
        assert os.path.isfile(
            args.q_table_path
        ), f"Q Table path {args.q_table_path} does not exist"
        evaluate(args.q_table_path, save=True)
