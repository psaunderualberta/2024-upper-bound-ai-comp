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

    def __init__(
        self,
        env_version=1,
        seed=1,
        path_difficulty=1.0,
        puddle_difficulty=1.0,
        stochastic=True,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.path_difficulty = path_difficulty
        self.puddle_difficulty = puddle_difficulty
        self.stochastic = stochastic
        env_setup = dict(self.kwargs, **self.get_env_setup())
        super().__init__(**env_setup)

    def reset(self, seed=None, options=None):
        if self.stochastic:
            new_seed = random.choice(self.valid_seeds)
            self.env_setup = dict(self.kwargs, **self.get_env_setup())
            super().__init__(**self.env_setup)
            return super().reset(new_seed)

        return super().reset()

    def get_env_setup(self):
        env_setup = None
        for env_version in self.valid_envs:
            json_file = os.path.join(
                file_location, "..", "env_configs", f"pw{env_version}.json"
            )

            json_file = os.path.abspath(json_file)
            assert os.path.isfile(json_file), f"File {json_file} does not exist!"
        
            with open(json_file) as f:
                new_env_setup = json.load(f)

            if env_setup is None:
                env_setup = new_env_setup
            else:
                env_setup["puddle_top_left"] += new_env_setup["puddle_top_left"]
                env_setup["puddle_width"] += new_env_setup["puddle_width"]

        # Remove puddles if difficulty is 0, otherwise scale the puddle width
        if self.puddle_difficulty == 0:
            env_setup["puddle_top_left"] = []
            env_setup["puddle_width"] = []
        else:
            for i, w in enumerate(env_setup["puddle_width"]):
                env_setup["puddle_width"][i] = list(map(lambda x: x * self.puddle_difficulty, w))

        # Shrink the path from 'path_difficulty' times the original start -> goal length
        env_setup["shrink_factor"] = 1 - self.path_difficulty

        if self.stochastic:
            env_setup["start"] = []

        env_setup["puddle_agg_func"] = "sum"

        return env_setup
