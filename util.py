import cv2
import os
import json

__file_location = os.path.abspath(os.path.dirname(__file__))
__path2envs = os.path.join(__file_location, "gym_puddle", "env_configs")

def combine_envs(envnos):
    # Combines the environments into a single image
    # Only adds the puddles on top of each other
    # Assumes the environments are the same size
    combined_env = {}
    for envno in envnos:
        with open(os.path.join(__path2envs, f"pw{envno}.json"), "r") as f:
            env = json.load(f)
        if "puddle_top_left" in combined_env:
            combined_env["puddle_top_left"] += env["puddle_top_left"]
            combined_env["puddle_width"] += env["puddle_width"]
        else:
            combined_env = env
    
    # Write
    with open(os.path.join(__path2envs, "pw-all.json"), "w") as f:
        json.dump(combined_env, f)




def visualize(frames, video_name = "video.mp4"):
    # Saves the frames as an mp4 video using cv2
    video_path = video_name
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()