"""
Simple cart that needs to reach a target (there is no pole).
"""

import logging
import gym
import gym.spaces
import gym.utils.seeding
from gym.envs.classic_control import rendering
import math
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


_MAX_FORCE_MAGNITUDE = 100.
_DELTA_T = 0.02  # Seconds per steps.
_NUM_FRAMES = 2  # Number of steps for each action.
_NUM_STABLE_SPEED = 5
_TARGET_LOCATIONS = [-0.8, 0.8]
_FRICTION = 0.1

_BATCH_SIZE = 32
_TRAIN_EVERY = 5
_OBSERVATION_SIZE = 2
_REPLAY_MEMORY_SIZE = 50 * 2
_PREDICT_BEFORE = 80
_LAYERS = [400, 300]
_NUM_CLASSES = 2
_FINAL_ITERATION = 79


class PrivateCartEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 50
  }

  def __init__(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.chosen_target = 0
    self.random_number = -1
    self.current_iteration = 0
    self.predicted = 0.5
    self.is_training = False

    high = np.array([
        np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
        np.finfo(np.float32).max  # Random number.
    ])
    self.observation_space = gym.spaces.Box(-high, high)
    self.action_space = gym.spaces.Box(-1, +1, (1,))  # Left-right.

    self._seed()
    self.reset()
    self.viewer = None

    # Just need to initialize the relevant attributes
    self._configure()

    # Create tensorflow model.
    self.session = tf.Session(graph=tf.Graph())
    with self.session.graph.as_default():
      self.CreateModel()
      tf.initialize_all_variables().run(session=self.session)
    self.session.graph.finalize()

    # Keep buffer of observations (enough for 10 episodes).
    self.replay_memory = UniformReplayMemory(_REPLAY_MEMORY_SIZE, (_OBSERVATION_SIZE,))

  def CreateModel(self):
    with tf.variable_scope('discriminator'):
      self.input_observation = tf.placeholder(tf.float32, shape=(None, _OBSERVATION_SIZE))
      previous_size = _OBSERVATION_SIZE
      previous_input = self.input_observation
      # Layers.
      for i, layer_size in enumerate(_LAYERS):
        with tf.variable_scope('layer_%d' % i):
          initializer = tf.random_uniform_initializer(minval=-1.0 / math.sqrt(previous_size),
                                                      maxval=1.0 / math.sqrt(previous_size))
          w = tf.get_variable('w', (previous_size, layer_size), initializer=initializer)
          b = tf.get_variable('b', (layer_size,), initializer=initializer)
          previous_input = tf.nn.xw_plus_b(previous_input, w, b)
          previous_input = tf.nn.relu(previous_input)
          previous_size = layer_size
      # Output prediction.
      with tf.variable_scope('output'):
        initializer = tf.random_uniform_initializer(minval=-1.0 / math.sqrt(previous_size),
                                                    maxval=1.0 / math.sqrt(previous_size))
        w = tf.get_variable('w', (previous_size, _NUM_CLASSES), initializer=initializer)
        b = tf.get_variable('b', (_NUM_CLASSES,), initializer=initializer)
        logits = tf.nn.xw_plus_b(previous_input, w, b)
        self.prediction = tf.nn.softmax(logits)

      # Loss.
      self.input_labels = tf.placeholder(tf.int32, shape=(None,))
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.input_labels)
      self.loss = tf.reduce_mean(loss)

      # Optimizer.
      optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
      gradients = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.apply_gradients(gradients)

  def TrainModel(self, observations, labels):
    loss, _ = self.session.run([self.loss, self.train_op], {self.input_observation: observations, self.input_labels: labels})

  def PredictModel(self, observation):
    observation = np.expand_dims(observation, axis=0)
    prediction = self.session.run(self.prediction, {self.input_observation: observation})
    prediction = prediction[0, ...]
    return prediction[1]

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    action = np.clip(action, -1., 1.)
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

    # Punish if the target wasn't reached.
    if self.current_iteration == self.spec.timestep_limit - 1:
      reward -= 100.

    # Predict (only on the first steps).
    if self.current_iteration < _PREDICT_BEFORE:
      self.predicted = self.PredictModel(np.array([x, x_dot]))
      # The more certain the prediction the less reward.
      reward_prediction = float(self.chosen_target * 2 - 1) * (self.predicted * 2. - 1.)
      reward_prediction = 0 if reward_prediction < 0 else reward_prediction
      reward -= reward_prediction * 10. * (1.0 - float(self.current_iteration) / float(self.spec.timestep_limit))

      # Train discriminator.
      if self.is_training:
        self.replay_memory.Add(np.array([x, x_dot]), self.chosen_target)
        if self.current_iteration % _TRAIN_EVERY == 0 and len(self.replay_memory) >= _BATCH_SIZE:
          observations, labels = self.replay_memory.Sample(_BATCH_SIZE)
          self.TrainModel(observations, labels)

    self.current_iteration += 1

    return np.array([dist_to_target, dist_to_other, x_dot, self.random_number]).ravel(), reward, done, {}

  def _reset(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.chosen_target = self.np_random.randint(2)
    self.random_number = 1. if self.random_number < 0. else -1.  # Either -1 or 1 for now.
    self.current_iteration = 0
    self.predicted = 0.5
    cart_position = self.np_random.uniform(low=-0.1, high=0.1)
    cart_velocity = self.np_random.uniform(low=-0.01, high=0.01)
    dist_to_target = cart_position - _TARGET_LOCATIONS[self.chosen_target]
    dist_to_other = cart_position - _TARGET_LOCATIONS[1 - self.chosen_target]
    self.state = (cart_position, cart_velocity)
    return np.array([dist_to_target, dist_to_other, cart_velocity, self.random_number]).ravel()

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
    scaled_prediction_location = (self.predicted * 2. - 1.) * _TARGET_LOCATIONS[1] * scale + screen_width / 2.0

    if self.viewer is None:
      self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
      # Central line.
      middle = rendering.Line((screen_width / 2, screen_height), (screen_width / 2, 0))
      middle.set_color(.8, .8, .8)
      self.viewer.add_geom(middle)
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
      prediction = rendering.make_circle(scaled_cart_height / 4)
      prediction.set_color(.0, .8, 0.)
      self.prediction_transform = rendering.Transform()
      prediction.add_attr(self.prediction_transform)
      self.viewer.add_geom(prediction)

    # Moving elements.
    self.target_transform.set_translation(scaled_target_location, scaled_cart_y + scaled_cart_height * 2)
    self.prediction_transform.set_translation(scaled_prediction_location, scaled_cart_y - scaled_cart_height * 2)
    x, _ = self.state
    scaled_cart_x = x * scale + screen_width / 2.0
    self.cart_transform.set_translation(scaled_cart_x, scaled_cart_y)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# Replay buffer.

class ReplayMemory(object):

  def __init__(self, max_capacity, observation_shape):
    self.size = 0
    self.max_capacity = max_capacity
    self.buffer_observations = np.empty((max_capacity,) + tuple(observation_shape), dtype=np.float32)
    self.buffer_labels = np.empty((max_capacity,), dtype=np.int32)
    self.current_index = 0

  def Add(self, observation, label):
    i = self.current_index
    self.buffer_observations[i, ...] = observation
    self.buffer_labels[i] = label
    self.current_index = int((i + 1) % self.max_capacity)
    self.size = int(max(i + 1, self.size))  # Maxes out at max_capacity.

  def __len__(self):
    return self.size

  def Sample(self, n):
    assert n <= self.size, 'Replay memory contains less than %d elements.' % n
    self.indices, weights = self.SampleIndicesAndWeights(n)
    return (self.buffer_observations[self.indices, ...],
            self.buffer_labels[self.indices])


class UniformReplayMemory(ReplayMemory):

  def __init__(self, max_capacity, observation_shape):
    super(UniformReplayMemory, self).__init__(max_capacity, observation_shape)
    self.uniform_weights = None

  def SampleIndicesAndWeights(self, n):
    if self.uniform_weights is None:
      self.uniform_weights = np.ones(n) / n
    return np.random.choice(self.size, n, replace=False), self.uniform_weights
