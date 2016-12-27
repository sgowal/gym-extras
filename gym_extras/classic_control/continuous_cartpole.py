"""
Classic cart-pole system originally implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
It has been modified to allow continuous input actions and gives additional rewards for staying centered.
"""

import logging
import math
import gym
import gym.spaces
import gym.utils.seeding
from gym.envs.classic_control import rendering
import numpy as np

logger = logging.getLogger(__name__)


# Physics.
_GRAVITY = 9.8
_CART_MASS = 1.
_POLE_MASS = .1
_TOTAL_MASS = _CART_MASS + _POLE_MASS
_POLE_HALF_LENGTH = .5
_POLE_MASS_TIMES_HALF_LENGTH = _POLE_HALF_LENGTH * _POLE_MASS
_MAX_FORCE_MAGNITUDE = 10.
_DELTA_T = 0.02  # Seconds per steps.

_FAILURE_ANGLE = 12. * math.pi / 180.
_FAILURE_X = 2.4


class ContinuousCartPoleEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 50
  }

  def __init__(self, target_location=0.):
    self.target_location = target_location
    self.reward_sigma = reward_sigma

    # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
    high = np.array([
        _FAILURE_X * 2, np.finfo(np.float32).max,
        _FAILURE_ANGLE * 2, np.finfo(np.float32).max
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
    x, x_dot, theta, theta_dot = state
    force = _MAX_FORCE_MAGNITUDE * action[0]
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + _POLE_MASS_TIMES_HALF_LENGTH * theta_dot * theta_dot * sintheta) / _TOTAL_MASS
    thetaacc = (_GRAVITY * sintheta - costheta * temp) / (_POLE_HALF_LENGTH * (4.0 / 3.0 - _POLE_MASS * costheta * costheta / _TOTAL_MASS))
    xacc = temp - _POLE_MASS_TIMES_HALF_LENGTH * thetaacc * costheta / _TOTAL_MASS
    x = x + _DELTA_T * x_dot
    x_dot = x_dot + _DELTA_T * xacc
    theta = theta + _DELTA_T * theta_dot
    theta_dot = theta_dot + _DELTA_T * thetaacc
    self.state = (x, x_dot, theta, theta_dot)

    # Early failure.
    done = (x < -_FAILURE_X
            or x > _FAILURE_X
            or theta < -_FAILURE_ANGLE
            or theta > _FAILURE_ANGLE)

    dist_penalty = 1. * np.abs(x - self.target_location) + theta ** 2.
    alive_bonus = 10.
    r = alive_bonus - dist_penalty

    if not done:
      reward = r
    elif self.steps_beyond_done is None:
      reward = r
    else:
      if self.steps_beyond_done == 0:
          logger.warn('You are calling "step()" even though this environment has already returned done = True. You should always call "reset()" once you receive "done = True" -- any further steps are undefined behavior.')
      self.steps_beyond_done += 1
      reward = 0.0

    return np.array(self.state), reward, done, {}

  def _reset(self):
    self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    self.steps_beyond_done = None
    return np.array(self.state)

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400

    world_width = _FAILURE_X * 2
    scale = screen_width / world_width
    scaled_cart_y = 100
    scaled_pole_width = 10.0
    scaled_pole_length = scale * _POLE_HALF_LENGTH * 2.
    scaled_cart_width = 50.0
    scaled_cart_height = 30.0
    scaled_target_location = self.target_location * scale + screen_width / 2.0

    if self.viewer is None:
      self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
      # Static non-moving elements.
      l, r, t, b = -scaled_cart_width / 2, scaled_cart_width / 2, scaled_cart_height / 2, -scaled_cart_height / 2
      axle_offset = scaled_cart_height / 4.0
      cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      self.cart_transform = rendering.Transform()
      cart.add_attr(self.cart_transform)
      self.viewer.add_geom(cart)
      l, r, t, b = -scaled_pole_width / 2, scaled_pole_width / 2, scaled_pole_length - scaled_pole_width / 2, -scaled_pole_width / 2
      pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      pole.set_color(.8, .6, .4)
      self.pole_transform = rendering.Transform(translation=(0, axle_offset))
      pole.add_attr(self.pole_transform)
      pole.add_attr(self.cart_transform)
      self.viewer.add_geom(pole)
      self.axle = rendering.make_circle(scaled_pole_width / 2)
      self.axle.add_attr(self.pole_transform)
      self.axle.add_attr(self.cart_transform)
      self.axle.set_color(.5, .5, .8)
      self.viewer.add_geom(self.axle)
      self.track = rendering.Line((0, scaled_cart_y), (screen_width, scaled_cart_y))
      self.track.set_color(0, 0, 0)
      self.viewer.add_geom(self.track)
      target_transform = rendering.Transform(translation=(scaled_target_location, scaled_cart_y + scaled_pole_length))
      target = rendering.make_circle(scaled_pole_width / 2)
      target.set_color(.8, 0., 0.)
      target.add_attr(target_transform)
      self.viewer.add_geom(target)

    # Moving elements.
    x = self.state
    scaled_cart_x = x[0] * scale + screen_width / 2.0
    self.cart_transform.set_translation(scaled_cart_x, scaled_cart_y)
    self.pole_transform.set_rotation(-x[2])

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')
