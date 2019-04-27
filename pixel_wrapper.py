import numpy as np
from gym.core import Wrapper
import skimage.transform

INITIAL_IMG_SIZE = 256

class PixelObservationWrapper(Wrapper):
    def __init__(self, env, stack=4, img_width=32):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.img_width = img_width
        
        self.stack = stack
        self.imgs = [np.zeros([3, self.img_width, self.img_width]) for _ in range(self.stack)]

    def render_obs(self, color_last=False):
        raw_img = self.env.render(mode='rgb_array', height=INITIAL_IMG_SIZE, width=INITIAL_IMG_SIZE)
        # import ipdb; ipdb.set_trace()
        resized = skimage.transform.resize(raw_img, (self.img_width, self.img_width))
        if color_last: return resized
        else: return resized.transpose([2, 0, 1])

    # normalize between -1 and 1
    def observation(self):
        # return (np.concatenate(self.imgs, axis=0) / 255. - 0.5) * 2

        return np.concatenate(self.imgs, axis=0)
        
        # obs = np.concatenate(self.imgs, axis=0)
        # obs.fill(0)
        # return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        img = self.render_obs()
        self.imgs.pop(0)
        self.imgs.append(img)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.imgs = [np.zeros([3, self.img_width, self.img_width]) for _ in range(self.stack - 1)] + [self.render_obs()]
        # for _ in range(self.stack):
        # self.step(np.zeros(self.action_space.shape))
        return self.observation()
