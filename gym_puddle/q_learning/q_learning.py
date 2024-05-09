import numpy as np

__NUM_ACTIONS = 4

### Taken from puddle_env.py
__ACTIONS = np.zeros((__NUM_ACTIONS, 2))
# [[-1, 0], [1, 0], [0, -1], [0, 1]]
for i in range(__NUM_ACTIONS):
    self.actions[i][i // 2] = i % 2 * 2 - 1

def q_create_table(granularity: float = 100):
    return np.random.random((granularity + 1, granularity + 1, __NUM_ACTIONS))

def q_get_action(q_table: np.ndarray, pos: np.ndarray) -> int:
    (x, y) = q_get_position(q_table, pos)
    return np.argmax(q_table[x, y, :])

def q_get_rollout(q_table: np.ndarray, exploration_factor: float, env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass

def q_learn_rollout(q_table: nd.ndarray, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, lr: float, gamma: float) -> None:
    for i in range(actions.shape[0]):
        (xt, yt) = states[i]
        (xtp1, ytp1) = states[i + 1]
        action = actions[i]
        q_table[xt, yt, action] = (1 - lr) * q_table[xt, yt, action] + lr * (rewards[i] + gamma * np.max(q_table[xtp1, ytp1, :]))

def q_get_bins(q_table: np.ndarray) -> np.ndarray:
    return np.linspace(0, 1, q_table.shape[0] - 1)

def q_get_position(q_table: np.ndarray, pos: np.ndarray) -> np.ndarray:
    bins = q_get_bins(q_table)
    return np.digitize(pos, bins) - 1

def q_save_data(q_table: np.ndarray, filepath: str) -> None:
    pass

def q_load_data(filepath) -> np.ndarray:
    pass