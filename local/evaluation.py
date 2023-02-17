import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


import random

from collections import deque

from agent import *
from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


policy = ActorCritic(256, 7, device).to(device)
policy.load_state_dict(torch.load(
    './models/agent_with_curiosity.pt', map_location=device))

policy.eval()



transforms = T.Compose(
    [T.Grayscale(), T.Resize((84, 84)), T.Normalize(0, 200)]
)

def preprocess(state):  # [240,256,3]
    state = np.transpose(state[40: 220, :], (2, 0, 1))
    state = torch.tensor(state.copy(), dtype=torch.float32)
    return transforms(state)



def standardize_array(array, mean, std):
  return (array - mean) / (std + 1e-8)

num_episodes = 1
rewards = []
states = deque(maxlen=4)
# real_states = []
eval_env = gym_super_mario_bros.make(
    'SuperMarioBros-1-1-v0', apply_api_compatibility=True)
eval_env = JoypadSpace(eval_env, SIMPLE_MOVEMENT)


state, _ = eval_env.reset()
real_states = [state.copy()]
with torch.no_grad():
    for e in range(num_episodes):
        state, _ = eval_env.reset()
        ep_reward = 0
        
        for _ in range(4):
            states.append(preprocess(state))
        
        done = False
        while not done:
            action = policy.act(states)

            state, reward, done, _, info = eval_env.step(action)
            ep_reward += reward/15
            real_states.append(state.copy())
            states.append(preprocess(state))
            
        rewards.append(ep_reward)
        print(f"Episode {e} reward: {ep_reward}")
    print(
        f"Mean reward across {num_episodes} episodes: {np.array(rewards).mean()}")


create_video(real_states)
