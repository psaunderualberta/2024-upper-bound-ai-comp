import numpy as np
from gym_puddle.env.puddle_env import is_done, get_reward, get_new_pos
from numba import njit

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
def q_create_table(
    granularity: float = 10
):
    tbl = np.zeros((granularity, granularity, __NUM_ACTIONS), dtype=np.float64)

    n = granularity
    for i in range(n):
        for j in range(n):
            pos = np.array([i, j], dtype=np.float64) / np.array([n, n], dtype=np.float64)
            for a in range(__NUM_ACTIONS):
                new_pos = get_new_pos(pos, __ACTIONS, a, __THRUST_NOISE)

                # The goal is always in the top-right, so
                # initalize the q table using distance to the
                # upper-left corner. 20 steps to go across whole map,
                # so normalize according to that
                tbl[i, j, a] = -np.linalg.norm((new_pos - __GOAL), ord=1) * 20

    return tbl


@njit
def q_get_action(q_table: np.ndarray, pos: np.ndarray) -> int:
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
    puddle_env_ids: np.ndarray
):
    pos = np.array([0.2, 0.4])  # Same start for every environment
    cum_reward = 0.0

    num_steps = 0
    while not is_done(pos, __GOAL, __GOAL_THRESHOLD):

        (yt, xt) = q_get_position(q_table, pos)
        if np.random.random() < exploration_factor:
            action = np.random.randint(__NUM_ACTIONS)
        else:
            action = q_get_action(q_table, pos)
        pos = get_new_pos(pos, __ACTIONS, action, __THRUST_NOISE)
        reward = get_reward(
            pos,
            __GOAL,
            __GOAL_THRESHOLD,
            puddle_top_left,
            puddle_width,
            puddle_env_ids,
        )

        (ytp1, xtp1) = q_get_position(q_table, pos)
        q_table[yt, xt, action] = (1 - lr) * q_table[yt, xt, action] + lr * (
            reward + gamma * np.max(q_table[ytp1, xtp1, :])
        )

        num_steps += 1
        cum_reward += reward

    return num_steps, cum_reward


@njit
def q_get_action(q_table: np.ndarray, pos: np.ndarray) -> int:
    (x, y) = q_get_position(q_table, pos)
    return np.argmax(q_table[x, y, :])


@njit
def q_get_position(q_table: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n = q_table.shape[0]
    y = min(int(pos[0] * n), n - 1)
    x = min(int(pos[1] * n), n - 1)
    return (y, x)

@njit
def q_upscale_table(q_table: np.ndarray) -> np.ndarray:
    (n, _, _) = q_table.shape
    new_q_table = np.zeros((n * 2, n * 2, __NUM_ACTIONS), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            new_q_table[2 * i, 2 * j, :] = q_table[i, j, :]
            new_q_table[2 * i + 1, 2 * j, :] = q_table[i, j, :]
            new_q_table[2 * i, 2 * j + 1, :] = q_table[i, j, :]
            new_q_table[2 * i + 1, 2 * j + 1, :] = q_table[i, j, :]
    return new_q_table

def q_save_data(q_table: np.ndarray, filepath: str) -> None:
    np.save(filepath, q_table)


def q_load_data(filepath) -> np.ndarray:
    return np.load(filepath)