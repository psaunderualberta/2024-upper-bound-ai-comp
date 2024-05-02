import gymnasium as gym
import gym_puddle

from stable_baselines3 import PPO
from stable_baselines3.dqn import MlpPolicy as DQNPolicy

from util import visualize

import os
import subprocess
from rl_zoo3.record_video import record_video as rl_zoo_record_video
from unittest.mock import patch
import sys


def record_video():
    # train the model, and save the trained model
    environment_name = "NoPuddleWorldStochastic-v0"

    prog_constants = [
        "record_video.py",
        "--algo", "ppo",
        "--env", environment_name,
        "-n", "1000",
        "-f", "logs/"
    ]

    dirs = os.listdir("logs/ppo")
    experiment_ids = [d.split("_")[-1] for d in dirs]
    print(experiment_ids)

    # Train the model without puddles
    save_path = ""
    for experiment_id in experiment_ids:

        # Append keyword arguments to the command
        prog_constants.extend(["--exp-id", experiment_id])
        print(" ".join(prog_constants))

        with patch.object(sys, "argv", prog_constants):
            rl_zoo_record_video()

if __name__ == "__main__":
    record_video()
