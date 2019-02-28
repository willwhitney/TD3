import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherSpinEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_spin.xml', 2)

    def step(self, a):
        reward_speed = self.sim.data.qvel[-1]
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward = reward_speed + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_speed=reward_speed, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        for _ in range(1000):
            qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            qvel[-1:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0:
                break
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        # use the xpos of the tip instead of the hinge qpos
        # because when the spinner spins, the qpos will go arbitrarily large
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.sim.data.qpos.flat[2:],
            # self.model.body_pos[-1, (-3, -1)],    # target's (x, z) coords
            self.sim.data.qvel.flat,
            self.sim.data.site_xpos[0, (0,2)], # cartesian position of tip of spinner
        ])

class ReacherSpinSparseEnv(ReacherSpinEnv):
    def step(self, a):
        # reward_dist = np.linalg.norm(vec)
        # reward_ctrl = - 0.1 * np.square(a).sum()
        reward = 1 if self.sim.data.qvel[-1] > 1 else 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict()
