import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

_NUM_STABLE_SPEED = 5
_TARGET_LOCATIONS = [-0.8, 0.8]


class ChoiceCartEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.chosen_target = 0
    utils.EzPickle.__init__(self)
    asset_file = os.path.join(os.path.dirname(__file__), 'assets', 'choice_cart.xml')
    mujoco_env.MujocoEnv.__init__(self, asset_file, 2)

  def _step(self, a):
    self.do_simulation(a, self.frame_skip)
    ob = self._get_obs()
    dist_to_target = np.abs(ob[0])
    self.previous_speeds[self.previous_speeds_index] = ob[1]
    self.previous_speeds_index = (self.previous_speeds_index + 1) % _NUM_STABLE_SPEED
    reward = - dist_to_target - np.square(a).sum()
    done = not np.isfinite(ob).all() or (
        dist_to_target < 0.05 and np.mean(np.abs(self.previous_speeds)) < 0.01)
    return ob, reward, done, {}

  def reset_model(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.chosen_target = self.np_random.randint(2)
    cart_position = self.np_random.uniform(low=-0.1, high=0.1)
    target_position = _TARGET_LOCATIONS[self.chosen_target]
    cart_velocity = self.np_random.uniform(low=-0.01, high=0.01)
    qpos = np.array([cart_position, target_position])
    qvel = np.array([cart_velocity, 0.])
    self.set_state(qpos, qvel)
    return self._get_obs()

  def _get_obs(self):
    dist_to_target = self.model.data.qpos[0] - _TARGET_LOCATIONS[self.chosen_target]
    dist_to_other = self.model.data.qpos[0] - _TARGET_LOCATIONS[1 - self.chosen_target]
    return np.array([dist_to_target, dist_to_other, self.model.data.qvel[0]]).ravel()

  def viewer_setup(self):
    v = self.viewer
    v.cam.trackbodyid = 0
    v.cam.distance = 3.
