import numpy as np
from gym_puddle.env.puddle_env import is_done, get_reward, get_new_pos
from numba import njit
import cv2

__NUM_ACTIONS = 4

# Constants in environments, may as well hardcode
__GOAL = np.array([1.0, 1.0])
__GOAL_THRESHOLD = 0.1
__THRUST = 0.05
__THRUST_NOISE = 0.01

### Taken from puddle_env.py
__ACTIONS = np.zeros((__NUM_ACTIONS, 2))
# [[-1, 0], [1, 0], [0, -1], [0, 1]] * __THRUST
for i in range(__NUM_ACTIONS):
    __ACTIONS[i][i // 2] = __THRUST * (i % 2 * 2 - 1)


@njit
def q_create_table(granularity: float = 10) -> np.ndarray:
    """Create a Q-table for the Q-learning algorithm.

    Args:
        granularity (float, optional): The number of cells along each dimension in the Q table. Defaults to 10.

    Returns:
        np.ndarray: A Q-table with shape (granularity, granularity, __NUM_ACTIONS)
    """
    tbl = np.zeros((granularity, granularity, __NUM_ACTIONS), dtype=np.float64)

    n = granularity
    for i in range(n):
        for j in range(n):
            pos = np.array([i, j], dtype=np.float64) / np.array(
                [n, n], dtype=np.float64
            )
            for a in range(__NUM_ACTIONS):
                new_pos = get_new_pos(pos, __ACTIONS, a, __THRUST_NOISE)

                # The goal is always in the top-right, so
                # initalize the q table using distance to the
                # upper-left corner. 20 steps to go across whole map,
                # so normalize according to that value.
                tbl[i, j, a] = -np.linalg.norm((new_pos - __GOAL), ord=1) * 20

    return tbl


@njit
def q_get_action(q_table: np.ndarray, pos: np.ndarray) -> int:
    """Get the action with the highest Q-value at a given position.

    Args:
        q_table (np.ndarray): The Q-table.
        pos (np.ndarray): The position in the puddle-world to get the action for.

    Returns:
        int: The action with the highest Q-value at the given position.
    """
    (y, x) = q_get_position(q_table, pos)
    return np.argmax(q_table[y, x, :])


@njit
def q_rollout(
    q_table: np.ndarray,
    exploration_factor: float,
    lr: float,
    gamma: float,
    puddle_top_left: np.ndarray,
    puddle_width: np.ndarray,
    puddle_env_ids: np.ndarray,
    start: np.ndarray = None,
) -> tuple[int, float]:
    """Rollout the Q-learning algorithm for a single episode.

    Args:
        q_table (np.ndarray): The Q-table to update.
        exploration_factor (float): The probability of taking a random action.
        lr (float): The learning rate.
        gamma (float): The discount factor.
        puddle_top_left (np.ndarray): The top-left corners of the puddles.
        puddle_width (np.ndarray): The widths of the puddles.
        puddle_env_ids (np.ndarray): The environment IDs of the puddles.
        start (np.ndarray, optional): . Defaults to None.

    Returns:
        tuple[int, float]: The number of steps taken and the cumulative reward.
    """
    if start is None:
        pos = np.random.uniform(0, 1, size=2)
    else:
        pos = start
    cum_reward = 0.0

    num_steps = 0
    while not is_done(pos, __GOAL, __GOAL_THRESHOLD):

        # Get the position in the Q-table
        (yt, xt) = q_get_position(q_table, pos)

        # Get the action to take, using epsilon-greedy
        if np.random.random() < exploration_factor:
            action = np.random.randint(__NUM_ACTIONS)
        else:
            action = q_get_action(q_table, pos)

        # Get the reward and the next position
        pos = get_new_pos(pos, __ACTIONS, action, __THRUST_NOISE)
        reward = get_reward(
            pos,
            __GOAL,
            __GOAL_THRESHOLD,
            puddle_top_left,
            puddle_width,
            puddle_env_ids,
        )

        # Get `pos` in the Q-table
        (ytp1, xtp1) = q_get_position(q_table, pos)

        # Different update rule if we are at the goal
        if is_done(pos, __GOAL, __GOAL_THRESHOLD):
            q_table[yt, xt, action] = (1 - lr) * q_table[yt, xt, action] + lr * reward
        else:
            q_table[yt, xt, action] = (1 - lr) * q_table[yt, xt, action] + lr * (
                reward + gamma * np.max(q_table[ytp1, xtp1, :])
            )

        # Increment trackers
        num_steps += 1
        cum_reward += reward

    # Reset the final position
    (yt, xt) = q_get_position(q_table, pos)
    q_table[yt, xt, :] = 0

    return num_steps, cum_reward


@njit
def q_get_position(q_table: np.ndarray, pos: np.ndarray) -> tuple[int, int]:
    """Get the position in the Q-table corresponding to a given position in the puddle-world.

    Args:
        q_table (np.ndarray):
        pos (np.ndarray): The position in the puddle-world.

    Returns:
        tuple[int, int]: The position in the Q-table.
    """
    n = q_table.shape[0]
    y = min(int(pos[0] * n), n - 1)
    x = min(int(pos[1] * n), n - 1)
    return (y, x)


def q_upscale_table(q_table: np.ndarray, upscale_factor: int) -> np.ndarray:
    """Upscale a Q-table by a given factor.

    Args:
        q_table (np.ndarray): The Q-table to upscale.
        upscale_factor (int): The factor to upscale the Q-table by.

    Returns:
        np.ndarray: The upscaled Q-table.
    """
    n = q_table.shape[0]
    return cv2.resize(
        q_table, (n * upscale_factor, n * upscale_factor), interpolation=cv2.INTER_LINEAR
    )


def q_save_data(q_table: np.ndarray, filepath: str) -> None:
    np.save(filepath, q_table)


def q_load_data(filepath) -> np.ndarray:
    return np.load(filepath)
