import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

_NUM_STABLE_SPEED = 5


class CartEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self, target_location=0.):
    self.target_location = target_location
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    utils.EzPickle.__init__(self)
    asset_file = os.path.join(os.path.dirname(__file__), 'assets', 'cart.xml')
    mujoco_env.MujocoEnv.__init__(self, asset_file, 2)

  def _step(self, a):
    self.do_simulation(a, self.frame_skip)
    ob = self._get_obs()
    dist_to_target = np.abs(ob[0] - self.target_location)
    self.previous_speeds[self.previous_speeds_index] = ob[1]
    self.previous_speeds_index = (self.previous_speeds_index + 1) % _NUM_STABLE_SPEED
    reward = - dist_to_target - np.square(a).sum()
    done = not np.isfinite(ob).all() or (
        dist_to_target < 0.05 and np.mean(np.abs(self.previous_speeds)) < 0.01)
    return ob, reward, done, {}

  def reset_model(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.8, high=0.8)
    qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
    self.set_state(qpos, qvel)
    return self._get_obs()

  def _get_obs(self):
    return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

  def viewer_setup(self):
    v = self.viewer
    v.cam.trackbodyid = 0
    v.cam.distance = v.model.stat.extent * 1.5
