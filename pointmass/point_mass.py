import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
import mujoco_py
from gym import error, spaces
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class LinearPointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        # self.model.nu = 2
        mujoco_env.MujocoEnv.__init__(self, dir_path + "/assets/point_mass.xml", 4)
        utils.EzPickle.__init__(self)

    def build_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, a):
        # action_cost = 1000 * np.linalg.norm(a)
        try:
            a = np.clip(a, self.action_space.low, self.action_space.high)
        except:
            self.build_action_space()
            a = np.clip(a, self.action_space.low, self.action_space.high)

        self.do_simulation(a, self.frame_skip)
        reward = - np.linalg.norm(self.sim.data.qpos)
        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def do_simulation(self, ctrl, n_frames):
        self.set_state(self.sim.data.qpos, ctrl)
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            self.sim.step()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    # def render(self, mode='human'):
    #     if mode == 'rgb_array':
    #         data = self.sim.render(550, 550)
    #         #self._get_viewer().render()
    #         # window size used for old mujoco-py:
    #         #width, height = 500, 500
    #         #data = self._get_viewer().read_pixels(width, height, depth=False)
    #         # original image is upside-down, so flip it
    #         return data[::-1, :, :]
    #     elif mode == 'human':
    #         self._get_viewer().render()

class SparsePointMassEnv(LinearPointMassEnv):
    def __init__(self, *args, **kwargs):
        self.startup = True
        super().__init__(*args, **kwargs)

    def step(self, a):
        # action_cost = 1000 * np.linalg.norm(a)
        try:
            a = np.clip(a, self.action_space.low, self.action_space.high)
        except:
            self.build_action_space()
            a = np.clip(a, self.action_space.low, self.action_space.high)

        self.do_simulation(a, self.frame_skip)
        target_dist = - np.linalg.norm(self.sim.data.qpos)
        reward_ctrl = - np.square(a).sum()

        reward = 10 if abs(target_dist) < 0.01 else 0
        reward += reward_ctrl
        done = (reward > 0) and not self.startup

        self.startup = False
        ob = self._get_obs()
        return ob, reward, done, {}

class SparsishPointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        # self.model.nu = 2
        self.startup = True
        mujoco_env.MujocoEnv.__init__(self, dir_path + "/assets/sparsish_point_mass.xml", 4)
        utils.EzPickle.__init__(self)

    def build_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, a):
        try:
            a = np.clip(a, self.action_space.low, self.action_space.high)
        except:
            self.build_action_space()
            a = np.clip(a, self.action_space.low, self.action_space.high)

        self.do_simulation(a, self.frame_skip)

        target_dist = np.linalg.norm(self.sim.data.qpos)
        # import ipdb; ipdb.set_trace()
        reward = 1 if target_dist < 0.02 else 0
        # reward = 1000 if target_dist < 0.02 else 0
        ob = self._get_obs()

        # done = (reward > 0) and not self.startup
        done = False

        self.startup = False
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

class OneDPointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        # self.model.nu = 2
        self.startup = True
        mujoco_env.MujocoEnv.__init__(self, dir_path + "/assets/1d_point_mass.xml", 4)
        utils.EzPickle.__init__(self)

    def build_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, a):
        try:
            a = np.clip(a, self.action_space.low, self.action_space.high)
        except:
            self.build_action_space()
            a = np.clip(a, self.action_space.low, self.action_space.high)

        self.do_simulation(a, self.frame_skip)

        target_dist = np.linalg.norm(self.sim.data.qpos)
        # import ipdb; ipdb.set_trace()
        reward = 1 if target_dist < 0.02 else 0
        # reward = 1000 if target_dist < 0.02 else 0
        ob = self._get_obs()

        # done = (reward > 0) and not self.startup
        done = False

        self.startup = False
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos,# + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel,# + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()


class LinearOneDPointMassEnv(OneDPointMassEnv):
    def do_simulation(self, ctrl, n_frames):
        self.set_state(self.sim.data.qpos, ctrl)
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            self.sim.step()


class ContactLinearPointMassEnv(LinearPointMassEnv):
    def __init__(self):
        # self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        # self.model.nu = 2
        mujoco_env.MujocoEnv.__init__(self, dir_path + "/assets/contact_point_mass.xml", 4)
        utils.EzPickle.__init__(self)


register(
    id='LinearPointMass-v0',
    entry_point='pointmass.point_mass:LinearPointMassEnv',
    max_episode_steps=100,
)
register(
    id='SparsePointMass-v0',
    entry_point='pointmass.point_mass:SparsePointMassEnv',
    max_episode_steps=100,
)
register(
    id='SparsishPointMass-v0',
    entry_point='pointmass.point_mass:SparsishPointMassEnv',
    max_episode_steps=1000,
)
register(
    id='ContactLinearPointMass-v0',
    entry_point='pointmass.point_mass:ContactLinearPointMassEnv',
    max_episode_steps=100,
)

register(
    id='1DPointMass-v0',
    entry_point='pointmass.point_mass:OneDPointMassEnv',
    max_episode_steps=100,
)
register(
    id='Linear1DPointMass-v0',
    entry_point='pointmass.point_mass:LinearOneDPointMassEnv',
    max_episode_steps=100,
)
