# Reviving Puddle World - Upper Bound (2024) AI Competition
Puddle World is an environment that got traction in the 1990s which was studied by Bryan and Moore (1995) and then later picked up by Rich Sutton in the same year. The agent starts at an initial state (denoted in red) in the Puddle World and the task for the agent is to navigate around the puddles (denoted in black) to reach the goal state (denoted in green). You can find more information about the environment in the paper "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"

This repository is an extension of the previous open-source implementation of the environment. This implementation is compatible with the gymnasium library, making it easy to interact with the environment.

<p align="center">
  <kbd>
    <img src='puddle_world.png'/>
  </kbd>
</p>

## Installation
Make a virtual env for your project

```python
python -m venv myenv
source myenv/bin/activate
```

Then navigate to the library directory and run this line in the library directory.

```python
pip install -e .
```

You can also find the details about the needed python and library versions in `setup.py`.

## Usage
```python
import gymnasium as gym
import gym_puddle # Don't forget this extra line!

env = gym.make('PuddleWorld-v0')
```

##  Configurations
Your task is to train an agent that can generalize well across different provided configurations of the environment. Each of these configurations feature different positions for puddles, which makes it challenging for the agent to find the most rewarding path to the goal.

You can find these configurations in the `env_configs` folder of the repository. In order to access each version of the environment, you can provide the `.json` file indicating the environment details, and intitialize the puddle world as mentioned in the  `getting_started.ipynb` Colab guide.

# More Details and Getting Started
For more details on how to get started with the environment, refer to `getting_started.ipynb` Colab file.

