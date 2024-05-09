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
# [[-1, 0], [1, 0], [0, -1], [0, 1]]
for i in range(__NUM_ACTIONS):
    __ACTIONS[i][i // 2] = __THRUST * (i % 2 * 2 - 1)

@njit
def vi_create_tables(
    puddle_top_left: np.ndarray,
    puddle_width: np.ndarray,
    puddle_agg_func: str,
    granularity: float = 10,
) -> tuple[np.ndarray, np.ndarray]:
    tbl = np.zeros((granularity + 1, granularity + 1, __NUM_ACTIONS), dtype=np.float64)

    # Distance from upper-left corner
    # 20 steps to go across whole map
    n = granularity + 1
    for i in range(n):
        for j in range(n):
            for a in range(__NUM_ACTIONS):
                pos = np.array([i / n, j / n])
                new_pos = get_new_pos(pos, __ACTIONS, a, __THRUST_NOISE)

                # The goal is always in the top-right, so
                # initalize the q table using distance to the
                # upper-left corner. 20 steps to go across whole map,
                # so normalize according to that
                tbl[i, j, a] = -np.linalg.norm((new_pos - __GOAL), ord=1) * 20

                # We also know the puddle positions, so incorporate that
                # knowledge as well :)
                reward = get_reward(
                    new_pos,
                    __GOAL,
                    __GOAL_THRESHOLD,
                    puddle_top_left,
                    puddle_width,
                    puddle_agg_func,
                )
                if reward == 0:
                    tbl[i, j, a] = 0
                elif reward != -1:
                    tbl[i, j, a] += reward

    return tbl

@njit
def vi_get_action(q_table: np.ndarray, pos: np.ndarray) -> int:
    (x, y) = q_get_position(q_table, pos)
    return np.argmax(q_table[x, y, :])

@njit
def vi_perform_update(q_table: np.ndarray, v_table: np.ndarray, puddle_top_left: np.ndarray, puddle_width: np.ndarray, puddle_agg_func: str) -> float:
    n = q_table.shape[0]
    num_iters = 500
    delta = 0

    for y in range(n):
        for x in range(n):
            pos = np.array([y / n, x / n])
            for action in range(__NUM_ACTIONS):
                new_val = 0

                for _ in range(num_iters):
                    new_pos = get_new_pos(pos, __ACTIONS, action, __THRUST_NOISE)
                    reward = get_reward(new_pos, __GOAL, __GOAL_THRESHOLD, puddle_top_left, puddle_width, puddle_agg_func)
                    (ny, nx) = vi_get_position(q_table, new_pos)
                    new_val += (reward + v_table[ny, nx]) / num_iters

                q_table[y, x, action] = new_val
            
            # print(np.max(q_table[y, x, :]), v_table[y, x])
            delta = max(delta, np.abs(np.max(q_table[y, x, :]) - v_table[y, x]))
            v_table[y, x] = np.max(q_table[y, x, :])

    return delta




    # while not is_done(pos, __GOAL, __GOAL_THRESHOLD):
    #     if np.random.random() < exploration_factor:
    #         action = np.random.randint(__NUM_ACTIONS)
    #     else:
    #         action = q_get_action(q_table, pos)
    #     pos = get_new_pos(pos, __ACTIONS, action, __THRUST_NOISE)
    #     reward = get_reward(pos, __GOAL, __GOAL_THRESHOLD, puddle_top_left, puddle_width, puddle_agg_func)

    #     states = np.append(states, pos)
    #     actions = np.append(actions, action)
    #     rewards = np.append(rewards, reward)

    return q_table, v_table

@njit    
def vi_get_bins(q_table: np.ndarray) -> np.ndarray:
    return np.linspace(0, 1, q_table.shape[0] - 1)

@njit
def vi_get_position(q_table: np.ndarray, pos: np.ndarray) -> np.ndarray:
    bins = vi_get_bins(q_table)
    return np.digitize(pos, bins) - 1

def vi_save_data(q_table: np.ndarray, filepath: str) -> None:
    np.save(filepath, q_table)

def vi_load_data(filepath) -> np.ndarray:
    return np.load(filepath)