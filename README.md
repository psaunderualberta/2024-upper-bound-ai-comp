# gym-puddle
Puddle World is an environment that got traction in the 1990s which was studied by Bryan and Moore (1995) and then later picked up by Rich Sutton in the same year. You can find more information about the environment in the paper "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"

The agent starts at an initial location or state in the Puddle World and the task for the agent is to navigate around the puddles (avoiding them) to reach the final destination or goal state.

You can find the open-source implementation of this environment in the gym-puddle github repository. This implementation is compatible with the gymnasium library, which makes it easy for you to interact with the environment.

<kbd>
  <img src='puddle_world.png'/>
</kbd>

## Installation
Make a virtual env for your project

```python
python -m venv venv
source venv/bin/activate
```

Then navigate to the library directory and run this line in the library directory.

```python
pip install -e .
```

## Usage
```python
import gymnasium as gym
import gym_puddle # Don't forget this extra line!

env = gym.make('PuddleWorld-v0')
```

##  Configurations
Your task is to train an agent that can generalize well across different provided configurations of the environment. Each of these configurations feature different positions for puddles, which makes it challenging for the agent to find the most rewarding path to the goal.

\\

You can find these configurations in the `config` folder of the repository. In order to access each version of the environment, you can provide the `.json` file indicating the environment details, and intitialize the puddle world as mentioned in the  `Getting_Started.ipynb` Colab guide.

# More Details and Getting Started
For more details on how to get started with the environment, refer to `Getting_Started.ipynb` Colab file.

