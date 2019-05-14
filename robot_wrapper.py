import numpy as np
from gym.core import ObservationWrapper
from gym import spaces

class RobotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space.spaces['observation']
        goal_space = env.observation_space.spaces['desired_goal']
        lows = np.concatenate([obs_space.low, goal_space.low])
        highs = np.concatenate([obs_space.high, goal_space.high])
        self.observation_space = spaces.Box(low=lows, high=highs)

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['desired_goal']])
