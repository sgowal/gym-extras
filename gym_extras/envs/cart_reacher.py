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


_MAX_FORCE_MAGNITUDE = 20.
_DELTA_T = 0.04  # Seconds per steps.


class CartReacherEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 50
  }

  def __init__(self, target_location=0.):
    self.target_location = target_location

    high = np.array([
        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
    ])
    self.action_space = gym.spaces.Box(-1, +1, (1,))  # Left-right.
    self.observation_space = gym.spaces.Box(-high, high)

    self._seed()
    self.reset()
    self.viewer = None
    self.steps_beyond_done = None

    # Just need to initialize the relevant attributes
    self._configure()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    action = np.maximum(action, -1.)
    action = np.minimum(action, 1.)
    assert self.action_space.contains(action), '%r (%s) invalid' % (action, type(action))

    # Run physics.
    state = self.state
    x, x_dot, _ = state
    force = _MAX_FORCE_MAGNITUDE * action[0]
    x = x + _DELTA_T * x_dot
    x_dot = x_dot + _DELTA_T * force

    dist_to_target = x - self.target_location
    self.state = (x, x_dot, dist_to_target)

    done = False
    r = - np.abs(dist_to_target) - np.square(force)

    return np.array(self.state), r, done, {}

  def _reset(self):
    self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(3,))
    self.state[-1] = self.state[0] - self.target_location
    return np.array(self.state)

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
    scaled_target_location = self.target_location * scale + screen_width / 2.0

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
      target_transform = rendering.Transform(translation=(scaled_target_location, scaled_cart_y + scaled_cart_height * 2))
      target = rendering.make_circle(scaled_cart_height / 2)
      target.set_color(.8, 0., 0.)
      target.add_attr(target_transform)
      self.viewer.add_geom(target)

    # Moving elements.
    x = self.state
    scaled_cart_x = x[0] * scale + screen_width / 2.0
    self.cart_transform.set_translation(scaled_cart_x, scaled_cart_y)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')
