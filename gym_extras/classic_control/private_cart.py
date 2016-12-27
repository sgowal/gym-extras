"""
Simple cart that needs to reach one of two targets (there is no pole).
The environment also has a discriminatory network that tries to identity to which target the agent will go.
"""

import logging
import gym
import gym.spaces
import gym.utils.seeding
from gym.envs.classic_control import rendering
import math
import numpy as np
import os
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_MAX_FORCE_MAGNITUDE = 100.
_DELTA_T = 0.02  # Seconds per steps.
_NUM_FRAMES = 2  # Number of steps for each action.
_NUM_STABLE_SPEED = 5
_TARGET_LOCATIONS = [-0.8, 0.8]
_FRICTION = 0.1


# Discrimatory network.
_BATCH_SIZE = 32
_TRAIN_EVERY_ITERATIONS = 5
_OBSERVATION_SIZE = 2
_ALLOWED_TO_PREDICT_BEFORE_ITERATION = 50
_REPLAY_MEMORY_SIZE = _ALLOWED_TO_PREDICT_BEFORE_ITERATION * 100
_LAYERS = [400, 300]  # [40, 30]?
_NUM_CLASSES = len(_TARGET_LOCATIONS)
_LEARNING_RATE = 1e-3
_DECAY_PREDICTION_REWARD = 0.95  # Reaches 0.1 after 80 iterations.
_DECAY_TARGET_REWARD = 1.05      # Reaches 50 after 80 iterations.
_PREDICTION_BONUS = 10.


class PrivateCartEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 50
  }

  def ResetEpisode(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.current_iteration = 1
    self.target_predicted = 0.5  # 0 means target 0, 1 means target 1.
    self.chosen_target_index = self.np_random.randint(2)
    # Reward discounts (as time goes on).
    self.prediction_reward_discount = 1.0
    self.target_reward_discount = 0.02

  def __init__(self, apply_discriminative_reward=True):
    self.episode_modulo = -1
    self.apply_discriminative_reward = apply_discriminative_reward
    self.is_training = False

    # Regular Gym environment setup with 5 observations.
    # Distance to correct target, distance to other target, speed, episode modulo and current time.
    high = np.array([
        np.finfo(np.float32).max, np.finfo(np.float32).max,
        np.finfo(np.float32).max, 1., 1.,
    ])
    self.observation_space = gym.spaces.Box(-high, high)
    self.action_space = gym.spaces.Box(-1, +1, (1,))  # Left-right.

    # Run all needed functions.
    self._seed()
    self.reset()
    self.viewer = None
    self._configure()

    # Discriminatory network.
    self.session = tf.Session(graph=tf.Graph())
    with self.session.graph.as_default():
      self.BuildDiscriminatoryNetwork()
      self.saver = tf.train.Saver(max_to_keep=1)
      tf.initialize_all_variables().run(session=self.session)
    self.session.graph.finalize()
    self.replay_memory = UniformReplayMemory(_REPLAY_MEMORY_SIZE, (_OBSERVATION_SIZE,))

  def Restore(self, checkpoint_directory):
    checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
    logger.info('Restoring discriminator from previous checkpoint: %s', checkpoint)

  def Save(self, checkpoint_directory, step):
    filename = self.saver.save(self.session, os.path.join(self.output_directory, 'discriminator.ckpt'))
    logger.info('Saving discriminator at %s', filename)

  def SetMode(self, is_training):
    self.is_training = is_training

  def HasSpecialFunctions(self):
    return True

  def BuildDiscriminatoryNetwork(self):
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
      optimizer = tf.train.AdamOptimizer(learning_rate=_LEARNING_RATE)
      gradients = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.apply_gradients(gradients)

  def TrainDiscriminator(self, observations, labels):
    loss, _ = self.session.run([self.loss, self.train_op],
                               {self.input_observation: observations, self.input_labels: labels})

  def Discriminate(self, observation):
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

    # Observation.
    dist_to_target = x - _TARGET_LOCATIONS[self.chosen_target_index]
    dist_to_other = x - _TARGET_LOCATIONS[1 - self.chosen_target_index]

    # Compute basic reward.
    abs_dist_to_target = np.abs(dist_to_target)
    self.previous_speeds[self.previous_speeds_index] = x_dot
    self.previous_speeds_index = (self.previous_speeds_index + 1) % _NUM_STABLE_SPEED
    reward = -abs_dist_to_target * self.target_reward_discount - np.square(action).sum()
    self.target_reward_discount *= _DECAY_TARGET_REWARD
    done = abs_dist_to_target < 0.05 and np.mean(np.abs(self.previous_speeds)) < 0.01

    # Try to discriminate which target was chosen from the current position and velocity.
    if self.current_iteration <= _ALLOWED_TO_PREDICT_BEFORE_ITERATION:
      self.target_predicted = self.Discriminate(np.array([x, x_dot]))
      # print self.target_predicted
      if self.apply_discriminative_reward:
        # The more certain the prediction the more reward the discriminator gets.
        reward_prediction = float(self.chosen_target_index * 2 - 1) * (self.target_predicted * 2. - 1.)
        reward_prediction = 0 if reward_prediction < 0 else reward_prediction  # Ignore bad prediction.
        reward -= reward_prediction * _PREDICTION_BONUS * self.prediction_reward_discount
        self.prediction_reward_discount *= _DECAY_PREDICTION_REWARD

      # Train discriminator.
      if self.is_training:
        self.replay_memory.Add(np.array([x, x_dot]), self.chosen_target_index)
        if self.current_iteration % _TRAIN_EVERY_ITERATIONS == 0 and len(self.replay_memory) >= _BATCH_SIZE:
          observations, labels = self.replay_memory.Sample(_BATCH_SIZE)
          self.TrainDiscriminator(observations, labels)

    # Keep track of time.
    time = float(self.current_iteration) / float(self.spec.timestep_limit) * 2. - 1.
    self.current_iteration += 1
    return np.array([dist_to_target, dist_to_other, x_dot, self.episode_modulo, time]).ravel(), reward, done, {}

  def _reset(self):
    self.ResetEpisode()
    self.episode_modulo = 1. if self.episode_modulo < 0. else -1.
    # Build first observation and state.
    cart_position = self.np_random.uniform(low=-0.1, high=0.1)
    cart_velocity = self.np_random.uniform(low=-0.01, high=0.01)
    dist_to_target = cart_position - _TARGET_LOCATIONS[self.chosen_target_index]
    dist_to_other = cart_position - _TARGET_LOCATIONS[1 - self.chosen_target_index]
    self.state = (cart_position, cart_velocity)
    return np.array([dist_to_target, dist_to_other, cart_velocity, self.episode_modulo, -1.]).ravel()

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
    scaled_target_location = _TARGET_LOCATIONS[self.chosen_target_index] * scale + screen_width / 2.0
    scaled_prediction_location = (self.target_predicted * (_TARGET_LOCATIONS[1] - _TARGET_LOCATIONS[0]) + _TARGET_LOCATIONS[0]) * scale + screen_width / 2.0

    if self.viewer is None:
      self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
      # Central line.
      middle = rendering.Line((screen_width / 2, screen_height), (screen_width / 2, 0))
      middle.set_color(.8, .8, .8)
      self.viewer.add_geom(middle)
      # Track.
      track = rendering.Line((0, scaled_cart_y), (screen_width, scaled_cart_y))
      track.set_color(0, 0, 0)
      self.viewer.add_geom(track)
      # Cart.
      l, r, t, b = -scaled_cart_width / 2, scaled_cart_width / 2, scaled_cart_height / 2, -scaled_cart_height / 2
      cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      self.cart_transform = rendering.Transform()
      cart.add_attr(self.cart_transform)
      self.viewer.add_geom(cart)
      # Target to reach.
      target = rendering.make_circle(scaled_cart_height / 4)
      target.set_color(.8, 0., 0.)
      self.target_transform = rendering.Transform()
      target.add_attr(self.target_transform)
      self.viewer.add_geom(target)
      # Current prediction.
      prediction = rendering.make_circle(scaled_cart_height / 4)
      prediction.set_color(.0, .8, 0.)
      self.prediction_transform = rendering.Transform()
      prediction.add_attr(self.prediction_transform)
      self.viewer.add_geom(prediction)

    # Move elements.
    self.target_transform.set_translation(scaled_target_location, scaled_cart_y + scaled_cart_height * 2)
    self.prediction_transform.set_translation(scaled_prediction_location, scaled_cart_y - scaled_cart_height * 2)
    x, _ = self.state
    scaled_cart_x = x * scale + screen_width / 2.0
    self.cart_transform.set_translation(scaled_cart_x, scaled_cart_y)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# Replay buffer.
# Copied from the ddpg package (https://github.com/sgowal/ddpg).

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
