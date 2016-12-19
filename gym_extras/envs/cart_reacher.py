"""
Simple cart that needs to reach a target (there is no pole).
"""

import logging
import math
import gym
import gym.spaces
import gym.utils.seeding
from gym.envs.classic_control import rendering
import numpy as np

logger = logging.getLogger(__name__)


_MAX_FORCE_MAGNITUDE = 100.
_DELTA_T = 0.02  # Seconds per steps.
_NUM_FRAMES = 2  # Number of steps for each action.
_NUM_STABLE_SPEED = 5
_TARGET_LOCATIONS = [-0.8, 0.8]
_FRICTION = 0.1


class CartReacherEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 50
  }

  def __init__(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.chosen_target = 0

    high = np.array([
        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
    ])
    self.observation_space = gym.spaces.Box(-high, high)
    self.action_space = gym.spaces.Box(-1, +1, (1,))  # Left-right.

    self._seed()
    self.reset()
    self.viewer = None

    # Just need to initialize the relevant attributes
    self._configure()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), '%r (%s) invalid' % (action, type(action))
    force = _MAX_FORCE_MAGNITUDE * action[0]

    # Run physics.
    x, x_dot = self.state
    for _ in xrange(_NUM_FRAMES):
      x += _DELTA_T * x_dot
      x = np.minimum(np.maximum(x, -2.), 2.)
      x_dot += _DELTA_T * force - x_dot * _FRICTION
      x_dot = np.minimum(np.maximum(x_dot, -20.), 20.)
    self.state = (x, x_dot)

    dist_to_target = np.abs(x - _TARGET_LOCATIONS[self.chosen_target])
    self.previous_speeds[self.previous_speeds_index] = x_dot
    self.previous_speeds_index = (self.previous_speeds_index + 1) % _NUM_STABLE_SPEED
    reward = - dist_to_target - np.square(action).sum()
    done = dist_to_target < 0.05 and np.mean(np.abs(self.previous_speeds)) < 0.01

    dist_to_target = x - _TARGET_LOCATIONS[self.chosen_target]
    dist_to_other = x - _TARGET_LOCATIONS[1 - self.chosen_target]
    return np.array([dist_to_target, dist_to_other, x_dot]).ravel(), reward, done, {}

  def _reset(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.chosen_target = self.np_random.randint(2)
    cart_position = self.np_random.uniform(low=-0.1, high=0.1)
    cart_velocity = self.np_random.uniform(low=-0.01, high=0.01)
    dist_to_target = cart_position - _TARGET_LOCATIONS[self.chosen_target]
    dist_to_other = cart_position - _TARGET_LOCATIONS[1 - self.chosen_target]
    self.state = (cart_position, cart_velocity)
    return np.array([dist_to_target, dist_to_other, cart_velocity]).ravel()

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400

    world_width = 2.4 * 2
    scale = screen_width / world_width
    scaled_cart_y = 100
    scaled_cart_width = 50.0
    scaled_cart_height = 30.0
    scaled_target_location = _TARGET_LOCATIONS[self.chosen_target] * scale + screen_width / 2.0

    if self.viewer is None:
      self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
      # Static non-moving elements.
      l, r, t, b = -scaled_cart_width / 2, scaled_cart_width / 2, scaled_cart_height / 2, -scaled_cart_height / 2
      cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      self.cart_transform = rendering.Transform()
      cart.add_attr(self.cart_transform)
      self.viewer.add_geom(cart)
      self.track = rendering.Line((0, scaled_cart_y), (screen_width, scaled_cart_y))
      self.track.set_color(0, 0, 0)
      self.viewer.add_geom(self.track)
      target = rendering.make_circle(scaled_cart_height / 4)
      target.set_color(.8, 0., 0.)
      self.target_transform = rendering.Transform()
      target.add_attr(self.target_transform)
      self.viewer.add_geom(target)

    # Moving elements.
    self.target_transform.set_translation(scaled_target_location, scaled_cart_y + scaled_cart_height * 2)
    x, _ = self.state
    scaled_cart_x = x * scale + screen_width / 2.0
    self.cart_transform.set_translation(scaled_cart_x, scaled_cart_y)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')
