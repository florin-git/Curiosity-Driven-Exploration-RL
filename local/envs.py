import numpy as np

import gym
from gym.spaces import Box

import torch
from torchvision import transforms as T

from collections import deque


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, truncated, info


class CutGrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation[40: 220, :], (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 200)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._pos = deque(maxlen = 8)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        if done:
            if info['flag_get']:
                reward += 1000.0
            else:
                reward -= 50.0

        return state, reward, done, truncated, info