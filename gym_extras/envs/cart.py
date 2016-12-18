import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env


class CartEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self, target_location=0.):
    self.target_location = target_location
    utils.EzPickle.__init__(self)
    asset_file = os.path.join(os.path.dirname(__file__), 'assets', 'cart.xml')
    mujoco_env.MujocoEnv.__init__(self, asset_file, 2)

  def _step(self, a):
    self.do_simulation(a, self.frame_skip)
    ob = self._get_obs()
    reward = - np.abs(ob[0] - self.target_location) - np.square(a).sum()
    notdone = np.isfinite(ob).all() and (np.abs(ob[0]) <= 1.)
    done = not notdone
    return ob, reward, done, {}

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.8, high=0.8)
    qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
    self.set_state(qpos, qvel)
    return self._get_obs()

  def _get_obs(self):
    return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

  def viewer_setup(self):
    v = self.viewer
    v.cam.trackbodyid = 0
    v.cam.distance = v.model.stat.extent
