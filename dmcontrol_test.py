import torch
import numpy as np

# import os
# os.environ["MUJOCO_GL"] = 'osmesa'

from dm_control import suite
from dm_control.suite.wrappers import pixels

import utils

env = suite.load(domain_name="humanoid", task_name="stand")
# import ipdb; ipdb.set_trace()
env = pixels.Wrapper(env)
spec = env.action_spec() 
time_step = env.reset() 
total_reward = 0.0
frames = [time_step.observation['pixels']]
for t in range(1000):
    print(t)
    action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)
    frames.append(time_step.observation['pixels'].copy())
    total_reward += time_step.reward

print("Total number of frames: {}".format(len(frames)))


# utils.save_gif('humanoid.mp4',
#                [torch.tensor(frame.copy()).float()/255 for frame in frames],
#                color_last=True)