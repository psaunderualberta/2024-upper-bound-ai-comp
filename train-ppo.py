import numpy as np
from rl_zoo3.train import train as rl_zoo_train
from unittest.mock import patch
import sys


def train():
    # train the model, and save the trained model
    environment_name = "NoPuddleWorldStochastic-v0"

    algorithm = "dqn"

    prog_constants = [
        "train.py",
        "--algo", algorithm,
        "--env", environment_name,
        "--conf-file", f"./config/{algorithm}.yml",
        "--eval-freq", "-1",
        "-n", "100000",
        # "-optimize", "--n-trials", "100", "--n-jobs", "3",
        # "--sampler", "tpe", "--pruner", "median"
    ]

    stochastic = True

    difficulties = np.linspace(0.1, 1, 10)

    # Train the model without puddles
    save_path = ""
    for path_difficulty in difficulties:

        if save_path:
            program = prog_constants + ["-i", f"{save_path}.zip"]
        else:
            program = prog_constants

        # Append keyword arguments to the command
        program = program + [
            "--env-kwargs",
            f"path_difficulty:{path_difficulty}",
            f"stochastic:{stochastic}",
            "puddle_difficulty:0.0",
        ]
        print(" ".join(program))

        with patch.object(sys, "argv", program):
            save_path = rl_zoo_train()

    # Train the model with puddles
    for puddle_difficulty in difficulties:

        if save_path:
            program = prog_constants + ["-i", f"{save_path}.zip"]
        else:
            program = prog_constants

        # Append keyword arguments to the command
        program = program + [
            "--env-kwargs",
            f"path_difficulty:1.0",
            f"stochastic:{stochastic}",
            f"puddle_difficulty:{puddle_difficulty}",
        ]
        print(" ".join(program))

        with patch.object(sys, "argv", program):
            save_path = rl_zoo_train()


if __name__ == "__main__":
    train()
