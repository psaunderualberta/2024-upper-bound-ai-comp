import numpy as np
from rl_zoo3.train import train as rl_zoo_train
from unittest.mock import patch
import sys
import os


file_location = os.path.abspath(os.path.dirname(__file__))

def optimize():
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
        "-optimize", "--n-trials", "100", "--n-jobs", "3",
        "--study-name", "test"
    ]

    stochastic = True

    path_difficulties = np.linspace(1, 1, 11)
    puddle_difficulties = np.linspace(0.8, 0.8, 1)

    # Train the model without puddles
    save_path = ""
    print(" ".join(prog_constants))
    exit()
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
        ]
        print(" ".join(program))

        with patch.object(sys, "argv", program):
            save_path = rl_zoo_train()


if __name__ == "__main__":
    optimize()