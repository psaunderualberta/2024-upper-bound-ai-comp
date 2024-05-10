"""
This module contains the implementation of the Value Iteration algorithm for the PuddleWorld environment.
"""

import numpy as np
from gym_puddle.env.puddle_env import is_done, get_reward, get_new_pos
from numba import njit, prange

__NUM_ACTIONS = 4

# Constants in environments, may as well hardcode
__GOAL = np.array([1.0, 1.0])
__GOAL_THRESHOLD = 0.1
__THRUST = 0.05
__THRUST_NOISE = 0.01

### Taken from puddle_env.py
__ACTIONS = np.zeros((__NUM_ACTIONS, 2))
# [[-1, 0], [1, 0], [0, -1], [0, 1]]
for i in range(__NUM_ACTIONS):
    __ACTIONS[i][i // 2] = __THRUST * (i % 2 * 2 - 1)


@njit
def vi_create_tables(
    puddle_top_left: np.ndarray,
    puddle_width: np.ndarray,
    puddle_agg_func: str,
    granularity: float = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((granularity + 1, granularity + 1, __NUM_ACTIONS), dtype=np.float64),
        np.zeros((granularity + 1, granularity + 1), dtype=np.float64) - 1,
        np.zeros((granularity + 1, granularity + 1, 2), dtype=np.float64),
    )


@njit
def vi_perform_update(
    q_table: np.ndarray,
    v_table: np.ndarray,
    r_table: np.ndarray,
    gamma: float,
    puddle_top_left: np.ndarray,
    puddle_width: np.ndarray,
    puddle_agg_func: str,
) -> float:
    n = q_table.shape[0]
    num_iters = 1
    delta = 0
    bin_len = 1 / n

    for i in range(n):  # Explicitly tell Numba to parallelize this loop
        y = n - 1 - i
        for x in range(n):
            x = n - 1 - x
            pos = np.array([y / n, x / n])
            pos += np.random.uniform(0, bin_len, 2)
            pos = np.clip(pos, 1e-5, 1 - 1e-5)

            if is_done(pos, __GOAL, __GOAL_THRESHOLD):
                v_table[y, x] = 0
                q_table[y, x, :] = 0
                continue

            for action in range(__NUM_ACTIONS):
                new_val = 0

                for _ in range(num_iters):
                    new_pos = get_new_pos(pos, __ACTIONS, action, 0)

                    reward = get_reward(
                        new_pos,
                        __GOAL,
                        __GOAL_THRESHOLD,
                        puddle_top_left,
                        puddle_width,
                        puddle_agg_func,
                    )
                    (ny, nx) = vi_get_position(q_table, new_pos)

                    # Running average of rewards
                    r_table[ny, nx][1] += 1
                    r_table[ny, nx][0] *= (r_table[ny, nx][1] - 1) / r_table[
                        ny, nx
                    ][1]
                    r_table[ny, nx][0] += reward / r_table[ny, nx][1]

                    new_val += r_table[ny, nx][0] + gamma * v_table[ny, nx]

                q_table[y, x, action] = new_val / num_iters
            # Update the value table, and keep track of the delta
            delta = max(delta, np.abs(np.max(q_table[y, x, :]) - v_table[y, x]))
            v_table[y, x] = np.max(q_table[y, x, :])

    # Check for NaNs and Infs
    assert np.isinf(v_table).sum() == 0 and np.isnan(v_table).sum() == 0

    return delta


@njit
def vi_get_action(q_table: np.ndarray, pos: np.ndarray) -> int:
    (x, y) = vi_get_position(q_table, pos)
    return np.argmax(q_table[x, y, :])


@njit
def vi_get_position(q_table: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n = q_table.shape[0] - 1
    y = int(pos[0] * n)
    x = int(pos[1] * n)
    return (y, x)


def vi_save_data(q_table: np.ndarray, filepath: str) -> None:
    np.save(filepath, q_table)


def vi_load_data(filepath) -> np.ndarray:
    return np.load(filepath)
