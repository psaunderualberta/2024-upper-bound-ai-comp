import numpy as np
from rl_zoo3.train import train as rl_zoo_train
from unittest.mock import patch
import sys


def train():
    # train the model, and save the trained model
    environment_name = "NoPuddleWorldStochastic-v0"

    print(sys.argv)
    prog_constants = [
        "train.py",
        "--algo", "ppo",
        "--env", environment_name,
        "--conf-file", "./config/ppo.yml",
        "--eval-freq", "-1",
        "-n", "50000",
    ]

    stochastic = True

    difficulties = np.linspace(1, 1, 10)

    # Train the model without puddles
    save_path = ""
    for path_difficulty in difficulties:

        if save_path:
            prog_constants.extend(["-i", f"{save_path}.zip"])

        # Append keyword arguments to the command
        prog_constants.extend(
            [
                "--env-kwargs",
                f"path_difficulty:{path_difficulty}",
                f"stochastic:{stochastic}",
                "puddle_difficulty:0.0",
            ]
        )
        print(" ".join(prog_constants))

        with patch.object(sys, "argv", prog_constants):
            save_path = rl_zoo_train()

    # Train the model with puddles
    for puddle_difficulty in difficulties:

        if save_path:
            prog_constants.extend(["-i", f"{save_path}.zip"])

        # Append keyword arguments to the command
        prog_constants.extend(
            [
                "--env-kwargs",
                f"path_difficulty:1.0",
                f"stochastic:{stochastic}",
                f"puddle_difficulty:{puddle_difficulty}",
            ]
        )
        print(" ".join(prog_constants))

        with patch.object(sys, "argv", prog_constants):
            save_path = rl_zoo_train()


if __name__ == "__main__":
    train()
