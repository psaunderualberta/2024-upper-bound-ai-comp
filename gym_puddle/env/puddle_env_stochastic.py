import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import copy
import pygame
from puddle_env.env.puddle_env import PuddleEnv
import os
import json
import numpy as np

__file_location = os.path.abspath(os.path.dirname(__file__))


class PuddleEnvStochastic(PuddleEnv):
    """
    An extension of the standard puddle env environment that allows for the location of puddles to
    change from underneath it. This is (hopefully) what we will use to train our submission.
    """

    valid_envs = np.arange(1, 6)  # There are 5 environment versions, labelled 1 - 5
    valid_seeds = np.arange(1, 101)  # 100 valid seeds, labelled 1 - 5

    def __init__(self, env_version=1, seed=1, stochastic=True):
        self.env_setup = self.get_env_setup(env_version, seed)
        super.__init__(**env_setup)

    def reset(self):
        if self.stochastic:
            new_env_version = np.random.choice(self.valid_envs)
            new_seed = np.random.choice(self.valid_seeds)
            self.env_setup = self.get_env_setup(new_env_version)
            super.__init__(**self.env_setup)
            super().reset(seed)
        else:
            super().reset()

    def get_env_setup(self, env_version):
        json_file = os.path.join(
            __file_location, "..", "env_configs", f"pw{env_version}.json"
        )
        json_file = os.path.abspath(json_file)
        assert os.path.isfile(json_file), f"File {json_file} does not exist!"

        with open(json_file) as f:
            env_setup = json.load(f)

        return env_setup
