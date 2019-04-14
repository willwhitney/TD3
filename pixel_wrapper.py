import numpy as np
from gym.core import Wrapper

IMG_SIZE = 128

class PixelObservationWrapper(Wrapper):
    def __init__(self, env, stack=3):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        
        self.stack = stack
        self.imgs = [np.zeros([3, IMG_SIZE, IMG_SIZE]) for _ in range(self.stack)]

    def render_obs(self):
        return self.env.render(mode='rgb_array', height=IMG_SIZE, width=IMG_SIZE).transpose([2, 1, 0])

    # normalize between -1 and 1
    def observation(self):
        return (np.concatenate(self.imgs, axis=0) / 255. - 0.5) * 2

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        img = self.render_obs()
        self.imgs.pop(0)
        self.imgs.append(img)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.imgs = [np.zeros([3, IMG_SIZE, IMG_SIZE]) for _ in range(self.stack)]
        return self.observation()
