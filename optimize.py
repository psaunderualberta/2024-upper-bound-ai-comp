import numpy as np
from rl_zoo3.train import train as rl_zoo_train
from unittest.mock import patch
import sys
import os

file_location = os.path.abspath(os.path.dirname(__file__))

def optimize():
    # train the model, and save the trained model
    environment_name = "NoPuddleWorldStochastic-v0"

    algorithm = "ppo"

    prog_constants = [
        "train.py",
        "--algo", algorithm,
        "--env", environment_name,
        "--conf-file", f"./config/{algorithm}.yml",
        "-n", "150000",
        "-optimize", "--n-trials", "200", "--n-jobs", "8"
    ]

    stochastic = True

    path_difficulties = np.linspace(1, 1, 1)
    puddle_difficulties = np.linspace(1, 1, 1)

    # Train the model without puddles
    save_path = ""
    for path_difficulty, puddle_difficulty in zip(path_difficulties, puddle_difficulties):

        if save_path:
            program = prog_constants + ["-i", f"{save_path}.zip"]
        else:
            program = prog_constants

        # Append keyword arguments to the command
        program = program + [
            "--env-kwargs",
            f"path_difficulty:{path_difficulty}",
            f"stochastic:{stochastic}",
            f"puddle_difficulty:{puddle_difficulty}",
            "puddle_agg_func:sum",
        ]
        print(" ".join(program))

        with patch.object(sys, "argv", program):
            save_path = rl_zoo_train()


if __name__ == "__main__":
    optimize()
