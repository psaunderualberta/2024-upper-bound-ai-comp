import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import copy
import pygame
from numba import njit


class PuddleEnv(gymnasium.Env):
    def __init__(
        self,
        start: list[float] = [0.2, 0.4],
        goal: list[float] = [1.0, 1.0],
        goal_threshold: float = 0.1,
        noise: float = 0.01,
        thrust: float = 0.05,
        puddle_top_left: list[list[float]] = [[0, 0.85], [0.35, 0.9]],
        puddle_width: list[list[float]] = [[0.55, 0.2], [0.2, 0.6]],
        render_mode: str = "rgb_array",
    ) -> None:
        """
        Initialize the PuddleEnv environment.

        Args:
            start (list[float]): Starting position of the agent.
            goal (list[float]): Goal position.
            goal_threshold (float): Threshold distance to consider the agent has reached the goal.
            noise (float): Magnitude of the noise added to the agent's actions.
            thrust (float): Magnitude of the agent's thrust.
            puddle_top_left (list[list[float]]): List of puddle top left positions.
            puddle_width (list[list[float]]): List of puddle width values.
        """

        self.start = np.array(start)
        self.goal = np.array(goal)

        self.goal_threshold = goal_threshold

        self.noise = noise
        self.thrust = thrust

        self.puddle_top_left = [np.array(top_left) for top_left in puddle_top_left]
        self.puddle_width = [np.array(width) for width in puddle_width]
        self.puddle_ids = np.zeros(len(self.puddle_top_left), dtype=np.int32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float64)

        self.actions = [np.zeros(2) for i in range(4)]

        for i in range(4):
            self.actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        self.num_steps = 0

        # Rendering
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 400
        self.min_reward = self.find_min_reward()
        self.heatmap = False

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform a step in the environment.

        Args:
            action (int): Action to be taken by the agent.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Tuple containing the new position, reward, done flag, trunc flag, and additional information.
        """
        self.num_steps += 1
        trunc = False  # we don't have a truncation condition for this environment
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        self.pos = get_new_pos(self.pos, self.actions, action, self.noise)

        reward = get_reward(
            self.pos,
            self.goal,
            self.goal_threshold,
            self.puddle_top_left,
            self.puddle_width,
            self.puddle_ids,
        )

        done = is_done(self.pos, self.goal, self.goal_threshold)

        return self.pos, reward, done, trunc, {}

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Seed value for the random number generator.
            options (dict): Additional options.

        Returns:
            tuple[np.ndarray, dict]: Tuple containing the initial position and additional information.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.num_steps = 0
        if len(self.start) == 0:
            self.pos = self.observation_space.sample()
            while is_done(
                self.pos, self.goal, self.goal_threshold
            ):  # make sure the start position is not too close to the goal
                self.pos = self.observation_space.sample()
        else:
            self.pos = copy.copy(self.start)
        return self.pos, {}

    def render(self) -> np.ndarray or None:  # type: ignore
        """
        Render the environment.

        Returns:
            numpy.ndarray or None: Rendered frame as an RGB array or None if the render mode is "human".
        """
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """
        Render a single frame of the environment.

        Returns:
            numpy.ndarray: Rendered frame as an RGB array.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # set teh window name
            pygame.display.set_caption("Puddle World")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if self.heatmap:
            # color the window as a heatmap based on the value of the reward at each pixel
            for i in range(self.window_size):
                for j in range(self.window_size):
                    pos = np.array([i / self.window_size, 1 - j / self.window_size])
                    reward = self._get_reward(pos)
                    if reward < -1:
                        max_reward = -1
                        color = int(
                            255
                            * (reward - self.min_reward)
                            / (max_reward - self.min_reward)
                        )
                        pygame.draw.rect(canvas, (255, color, 0), (i, j, 1, 1))

        # Draw the goal
        goal_pos = (
            int(self.goal[0] * self.window_size) - 10,
            self.window_size - int(self.goal[1] * self.window_size) + 10,
        )
        pygame.draw.circle(canvas, (0, 255, 0), goal_pos, 10)

        # Draw the puddles
        for top_left, wid in zip(self.puddle_top_left, self.puddle_width):
            puddle_pos = (
                int(top_left[0] * self.window_size),
                self.window_size - int(top_left[1] * self.window_size),
            )
            puddle_size = (
                int(wid[0] * self.window_size),
                int(wid[1] * self.window_size),
            )
            pygame.draw.ellipse(canvas, (0, 0, 0), (puddle_pos, puddle_size))

        # Draw the agent
        agent_pos = (
            int(self.pos[0] * self.window_size),
            self.window_size - int(self.pos[1] * self.window_size),
        )
        pygame.draw.circle(canvas, (255, 0, 0), agent_pos, 5)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def close(self) -> None:
        """
        Close the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def find_min_reward(self) -> float:
        """
        Find the minimum reward value in the environment.

        Returns:
            float: Minimum reward value.
        """
        min_reward = float("inf")
        for i in range(100):
            for j in range(100):
                pos = np.array([i / 100, j / 100])
                reward = get_reward(
                    pos,
                    self.goal,
                    self.goal_threshold,
                    self.puddle_top_left,
                    self.puddle_width,
                    self.puddle_ids,
                )
                if reward < min_reward:
                    min_reward = reward

        return min_reward

    def close(self) -> None:
        """
        Close the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


@njit
def get_reward(
    pos: np.ndarray,
    goal: np.ndarray,
    goal_threshold: np.ndarray,
    puddle_top_left: np.ndarray,
    puddle_width: np.ndarray,
    puddle_ids: np.ndarray,
) -> float:
    reward_puddles = np.zeros((np.unique(puddle_ids).size,), dtype=np.float64)
    for top_left, wid, i in zip(puddle_top_left, puddle_width, puddle_ids):
        if (
            top_left[0] <= pos[0] <= top_left[0] + wid[0]
            and top_left[1] - wid[1] <= pos[1] <= top_left[1]
        ):
            # Calculate the distance from the nearest edge of the puddle to the agent
            dist_to_edge = np.array(
                [
                    np.abs(pos[0] - top_left[0]),
                    np.abs(top_left[0] + wid[0] - pos[0]),
                    np.abs(pos[1] - top_left[1]),
                    np.abs(top_left[1] - wid[1] - pos[1]),
                ]
            ).max()

            reward_puddles[i] = min(reward_puddles[i], -400 * dist_to_edge)
    
    in_puddle = np.any(reward_puddles < 0)

    if not in_puddle and is_done(pos, goal, goal_threshold):
        return 0  # If the agent is in the goal, return 0
    elif not in_puddle:
        return -1  # -1 for each timestep
    else:
        return np.mean(reward_puddles)


@njit
def get_new_pos(pos, actions, action, noise):
    added_noise = np.random.normal(loc=0.0, scale=noise, size=(2,))
    return np.clip(pos + actions[action] + added_noise, 0, 1)


@njit
def is_done(pos: np.ndarray, goal: np.ndarray, goal_threshold: float) -> bool:
    return np.linalg.norm((pos - goal), ord=1) < goal_threshold
