import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import copy
import pygame
from gym_puddle.env.puddle_env import PuddleEnv
import os
import json
import random

file_location = os.path.abspath(os.path.dirname(__file__))


class NoPuddleEnvStochastic(PuddleEnv):
    """
    An extension of the standard puddle env environment that allows for the location of puddles to
    change from underneath it. This is (hopefully) what we will use to train our submission.
    """

    valid_envs = list(range(1, 6))  # There are 5 environment versions, labelled 1 - 5
    valid_seeds = list(range(1, 101))  # 100 valid seeds, labelled 1 - 5

    def __init__(self, env_version=1, seed=1, stochastic=False, **kwargs):
        self.kwargs = kwargs
        env_setup = dict(self.kwargs, **self.get_env_setup(env_version))
        super().__init__(**env_setup)
        self.stochastic = stochastic

    def reset(self, seed = None, options = None):
        if self.stochastic:
            new_env_version = random.choice(self.valid_envs)
            new_seed = random.choice(self.valid_seeds)
            env_setup = dict(self.kwargs, **self.get_env_setup(new_env_version))
            super().__init__(**self.env_setup)
            return super().reset(new_seed)
        else:
            return super().reset()

    def get_env_setup(self, env_version):
        json_file = os.path.join(
            file_location, "..", "env_configs", f"pw{env_version}.json"
        )
        json_file = os.path.abspath(json_file)
        assert os.path.isfile(json_file), f"File {json_file} does not exist!"

        with open(json_file) as f:
            env_setup = json.load(f)

        # Remove puddles
        env_setup["puddle_top_left"] = []
        env_setup["puddle_width"] = []

        return env_setup
