import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import copy
import pygame


class PuddleEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        start=[0.2, 0.4],
        goal=[1.0, 1.0],
        goal_threshold=0.1,
        noise=0.025,
        thrust=0.05,
        puddle_center=[[0.3, 0.6], [0.4, 0.5], [0.8, 0.9]],
        puddle_width=[[0.1, 0.03], [0.03, 0.1], [0.03, 0.1]],
    ):
        self.start = np.array(start)
        self.goal = np.array(goal)

        self.goal_threshold = goal_threshold

        self.noise = noise
        self.thrust = thrust

        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float64)

        self.actions = [np.zeros(2) for i in range(5)]

        for i in range(4):
            self.actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        self.num_steps = 0

        # Rendering
        self.render_mode = "human"
        self.window = None
        self.clock = None
        self.window_size = 400
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.min_reward = self.find_min_reward()
        self.heatmap = False

    def step(self, action):
        # if number of steps taken is more than 1e5, then trunc is set to True
        self.num_steps += 1
        if self.num_steps > 1e5:
            trunc = True
        trunc = False  # TODO: handle that later
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        noise = self.np_random.normal(loc=0.0, scale=self.noise, size=(2,))
        self.pos += self.actions[action] + noise
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        return self.pos, reward, done, trunc, {}

    def _get_reward(self, pos):
        reward = float("inf")  # Initialize reward with a large positive value
        reward_puddles = []
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            if cen[0] <= pos[0] <= cen[0] + wid[0] and cen[1] - wid[1]<= pos[1] <= cen[1]:
                # Calculate the distance from the nearest edge of the puddle to the agent
                dist_to_edge = max(
                    abs(pos[0] - cen[0]),
                    abs(cen[0] + wid[0] - pos[0]),
                    abs(pos[1] - cen[1]),
                    abs(cen[1] - wid[1] - pos[1])
                )
                reward_puddle = min(reward, -400 * dist_to_edge)
                reward_puddles.append(reward_puddle)
        if reward_puddles == [] and np.linalg.norm((pos - self.goal), ord=1) < self.goal_threshold:
            return 0
        elif reward_puddles == []:
            return -1
        #print(reward_puddles)
        return min(reward_puddles)

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu) ** 2) / (2.0 * sig**2)) / (sig * np.sqrt(2.0 * np.pi))

    def reset(self, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        if self.start is None:
            self.pos = self.observation_space.sample()
            while np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold: #make sure the start position is not too close to the goal
                self.pos = self.observation_space.sample()
        else:
            self.pos = copy.copy(self.start)
        return self.pos, {}

    def render(self):
        self.render_mode = "human"
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
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

        # Draw the goal, note that pygame coordination starts from top left but I want the center to be at down left
        goal_pos = (
            int(self.goal[0] * self.window_size) - 10,
            self.window_size - int(self.goal[1] * self.window_size) + 10,
        )
        pygame.draw.circle(canvas, (0, 255, 0), goal_pos, 10)



        # Draw the agent, note that pygame coordination starts from top left but I want the center to be at down left
        agent_pos = (
            int(self.pos[0] * self.window_size),
            self.window_size - int(self.pos[1] * self.window_size),
        )
        pygame.draw.circle(canvas, (255, 0, 0), agent_pos, 5)

        # Draw the puddle, note that pygame coordination starts from top left but I want the center to be at down left
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            puddle_pos = (
                int(cen[0] * self.window_size),
                self.window_size - int(cen[1] * self.window_size),
            )
            puddle_size = (int(wid[0] * self.window_size), int(wid[1] * self.window_size))
            pygame.draw.ellipse(canvas, (0, 0, 0), (puddle_pos, puddle_size))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))
        return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def find_min_reward(self):
        min_reward = float("inf")
        for i in range(100):
            for j in range(100):
                pos = np.array([i / 100, j / 100])
                reward = self._get_reward(pos)
                if reward < min_reward:
                    min_reward = reward

        return min_reward
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
