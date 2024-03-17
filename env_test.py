import gymnasium
import numpy as np
import time
import gym_puddle # register the environment. You need to import the environment


from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy as DQNPolicy
import cv2
import json


def play_by_human(env):
    # #play by user
    total_reward = 0
    frames = []
    obs, info = env.reset()
    print(obs, info)
    while True:
        action = int(input("enter action: "))
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        print("obs", obs)
        print("reward", reward)
        print("done", done)
        print("info", info)
        frames.append(env.render())
        time.sleep(0.1)
        if done:
            print("you did it!")
            print("total reward:", total_reward)
            break
    env.close()
    return frames


def train_by_DQN(env):
    dqn_model = DQN(DQNPolicy, env, verbose=1, exploration_final_eps=0.01, exploration_initial_eps=0.3, exploration_fraction=0.9, buffer_size=int(1e3))
    dqn_model.learn(total_timesteps=int(1e6))
    dqn_model.save("dqn_model")
    dqn_model = DQN.load("dqn_model")

    obs, info = env.reset()
    # Create an empty list to store the frames
    frames = []

    while True:
        action, _states = dqn_model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
        print("obs", obs)
        print("reward", reward)
        print("done", done)
        print("info", info)
        frames.append(
            env.render()
        )  # Append the rendered frame to the list
        print(frames[-1].shape)

        time.sleep(0.1)
        if done:
            print("you did it!")
            break
    return frames


def visualize(frames):
    # Save the frames as an mp4 video using cv2
    video_path = "video.mp4"
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def print_info(env):
    print("env", env)
    print("start", env.get_wrapper_attr("start"))
    print("env.goal:", env.get_wrapper_attr("goal"))
    print("env.goal_threshold:", env.get_wrapper_attr("goal_threshold"))
    print("env.noise:", env.get_wrapper_attr("noise"))
    print("env.thrust:", env.get_wrapper_attr("thrust"))
    print("env.puddle_center:", env.get_wrapper_attr("puddle_top_left"))
    print("env.puddle_width:", env.get_wrapper_attr("puddle_width"))
    print("env.action_space:", env.get_wrapper_attr("action_space"))
    print("env.observation_space:", env.get_wrapper_attr("observation_space"))
    print("env.actions:", env.get_wrapper_attr("actions"))
    print("env.actions type:", type(env.get_wrapper_attr("actions")))
    print("env.actions elements:", env.get_wrapper_attr("start")[1])
    # check if the action space is discrete and the observation space is continuous
    print(
        "Action space is discrete:",
        isinstance(env.action_space, gymnasium.spaces.Discrete),
    )
    print(
        "Observation space is continuous:",
        isinstance(env.observation_space, gymnasium.spaces.Box),
    )


if __name__ == "__main__":
    json_file = f"gym_puddle/env_setups/paper_compatible.json"
    with open(json_file) as f:
        env_setup = json.load(f)
    env = gymnasium.make(
        "PuddleWorld-v0",
        # start=env_setup["start"],
        # goal=env_setup["goal"],
        # goal_threshold=env_setup["goal_threshold"],
        # noise=env_setup["noise"],
        # thrust=env_setup["thrust"],
        # puddle_top_left=env_setup["puddle_center"],
        # puddle_width=env_setup["puddle_width"],
    )
    # print_info(env)
    # frames = train_by_DQN(env)
    # visualize(frames)
    # print(env.unwrapped.find_min_reward())
    env.reset()
    env.render()
    time.sleep(5)
    env.close()
    # frames = play_by_human(env)
    # visualize(frames)
