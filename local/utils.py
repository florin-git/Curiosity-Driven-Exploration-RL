import numpy as np

from IPython.display import HTML
from base64 import b64encode
import cv2
import os, random

from collections import deque

from torch.multiprocessing import Process

import gym_super_mario_bros
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from envs import *


global_epochs = 10
n_step = 2  # step del rollout

def create_environment():
  en = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
  en = JoypadSpace(en, SIMPLE_MOVEMENT)
  en = CustomReward(en)
  en = SkipFrame(en, skip = 4)
  en = CutGrayScaleObservation(en)
  en = ResizeObservation(en, shape = 84)
  en = FrameStack(en, num_stack = 4)
  return en



def compute_target_advantage(reward, done, value, gamma, n_step, num_workers):
    discounted_return = np.empty([num_workers, n_step])

    # Take the value from the last state (s_{t_max})
    running_add = value[:, -1]
    for t in range(n_step - 1, -1, -1):
        running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
        discounted_return[:, t] = running_add

    # For Actor
    adv = discounted_return - value[:, :-1]

    return discounted_return.reshape(-1), adv.reshape(-1)


def create_video(images, file_name="output"):
    # Remove old videos
    if os.path.exists(f"./video/{file_name}.mp4"):
        os.remove(f"./video/{file_name}.mp4")
    if os.path.exists(f"./video/{file_name}_compressed.mp4"):
        os.remove(f"./video/{file_name}_compressed.mp4")
    
    # set the fourcc codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # get the height and width of the first image
    height, width, _ = images[0].shape

    # create a VideoWriter object
    fps = 20
    out = cv2.VideoWriter(
        f"./video/{file_name}.mp4", fourcc, float(fps), (width, height))

    # write each image to the video file
    for img in images:
        # convert image to BGR color space
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    # release the VideoWriter object
    out.release()

    # Compressed video path
    compressed_path = f"./video/{file_name}_compressed.mp4"
    os.system(
        f"ffmpeg -i ./video/{file_name}.mp4 -vcodec libx264 {compressed_path}")

def show_video(compressed_path="./video/output_compressed.mp4"):
    mp4 = open(compressed_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url)