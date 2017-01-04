"""
Simple cart that needs to reach one of two targets (there is no pole).
The environment also has a discriminatory network that tries to identity to which target the agent will go.
"""

import logging
import gym
from gym.envs.classic_control import rendering
import gym.spaces
import gym.utils.seeding
import math
import numpy as np
import os
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Physics.
_MAX_FORCE_MAGNITUDE = 100.
_DELTA_T = 0.02  # Seconds per steps.
_NUM_FRAMES = 2  # Number of steps for each action.
_NUM_STABLE_SPEED = 5
_TARGET_LOCATIONS = [-0.8, 0.8]
_FRICTION = 0.1

# Give separate reward for motion actions.
_MOTION_MULTIPLIER = 1.

# Discrimatory network.
_OBSERVATION_SIZE = 2  # Position and speed.
_DEFAULT_LAYERS = [10]  # Small default network to avoid overfitting.
_NUM_CLASSES = len(_TARGET_LOCATIONS)


class PrivateCartEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 50
  }

  def __init__(self):
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

  def SetOptions(self, options):
    self.options = options
    if not self.options.layer_size:
      self.options.layer_size.extend(_DEFAULT_LAYERS)
    # Discriminatory network.
    self.session = tf.Session(graph=tf.Graph())
    with self.session.graph.as_default():
      self._CreateModel()
      self.saver = tf.train.Saver(max_to_keep=1)
      tf.initialize_all_variables().run(session=self.session)
    self.session.graph.finalize()
    self.replay_memory = UniformReplayMemory(self.options.replay_memory_size, (_OBSERVATION_SIZE,))

  def Restore(self, checkpoint_directory):
    checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
    logger.info('Restoring discriminator from previous checkpoint: %s', checkpoint)

  def Save(self, checkpoint_directory, step):
    filename = self.saver.save(self.session, os.path.join(checkpoint_directory, 'discriminator.ckpt'))
    logger.info('Saving discriminator at %s', filename)

  def SetTrainingMode(self, is_training):
    self.is_training = is_training

  def IsPrivate(self):
    return True

  def GetState(self):
    return np.array(self.state)

  def GetChosenTarget(self):
    return self.chosen_target_index

  def TargetLabels(self):
    return ('Left', 'Right')

  def _CreateModel(self):
    parameters = self._DiscriminatorParameters()
    parameters_target, update_target = _PropagateToTargetNetwork(parameters, 1. - self.options.tau)

    self.input_observation = tf.placeholder(tf.float32, shape=(None, _OBSERVATION_SIZE))
    logits = self._DiscriminatorNetwork(self.input_observation, parameters)

    # Train.
    self.input_labels = tf.placeholder(tf.int32, shape=(None,))
    self.input_weights = tf.placeholder(tf.float32, shape=(None,))
    self.loss = tf.reduce_mean(self.input_weights * tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.input_labels))
    self.loss += tf.add_n([self.options.weight_decay * tf.nn.l2_loss(p) for p in parameters])
    optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)
    gradients = optimizer.compute_gradients(self.loss)
    train_op = optimizer.apply_gradients(gradients)
    with tf.control_dependencies([train_op]):
      self.train_op = tf.group(update_target)

    # Predict (use target network).
    self.single_input_observation = tf.placeholder(tf.float32, shape=(1, _OBSERVATION_SIZE))
    logits = self._DiscriminatorNetwork(self.single_input_observation, parameters_target)
    self.prediction = tf.nn.softmax(logits)

  def _DiscriminatorParameters(self):
    params = []
    with tf.variable_scope('discriminator'):
      previous_size = _OBSERVATION_SIZE
      # Layers.
      for i, layer_size in enumerate(self.options.layer_size):
        with tf.variable_scope('layer_%d' % i):
          initializer = tf.random_uniform_initializer(minval=-1.0 / math.sqrt(previous_size),
                                                      maxval=1.0 / math.sqrt(previous_size))
          w = tf.get_variable('w', (previous_size, layer_size), initializer=initializer)
          b = tf.get_variable('b', (layer_size,), initializer=initializer)
          params.extend([w, b])
          previous_size = layer_size
      # Output prediction.
      with tf.variable_scope('output'):
        initializer = tf.random_uniform_initializer(minval=-1.0 / math.sqrt(previous_size),
                                                    maxval=1.0 / math.sqrt(previous_size))
        w = tf.get_variable('w', (previous_size, _NUM_CLASSES), initializer=initializer)
        b = tf.get_variable('b', (_NUM_CLASSES,), initializer=initializer)
        params.extend([w, b])
    return params

  def _DiscriminatorNetwork(self, observation, params):
    index = 0
    with tf.variable_scope('discriminator'):
      previous_input = observation
      # Layers.
      for i, layer_size in enumerate(self.options.layer_size):
        with tf.variable_scope('layer_%d' % i):
          w = params[index]
          b = params[index + 1]
          index += 2
          previous_input = tf.nn.xw_plus_b(previous_input, w, b)
          previous_input = tf.nn.relu(previous_input)
      # Output prediction.
      with tf.variable_scope('output'):
        w = params[index]
        b = params[index + 1]
        index += 2
        logits = tf.nn.xw_plus_b(previous_input, w, b)
        return logits

  def _TrainDiscriminator(self, observations, labels, weights):
    self.session.run(self.train_op, {self.input_observation: observations, self.input_labels: labels, self.input_weights: weights})

  def _Discriminate(self, observation):
    observation = np.expand_dims(observation, axis=0)
    prediction = self.session.run(self.prediction, {self.single_input_observation: observation})
    prediction = prediction[0, ...]
    return prediction[1]

  def _ResetEpisode(self):
    self.previous_speeds = np.ones(_NUM_STABLE_SPEED)
    self.previous_speeds_index = 0
    self.current_iteration = 1
    self.target_predicted = 0.5  # 0 means target 0, 1 means target 1.
    self.chosen_target_index = self.np_random.randint(2)
    self.episode_modulo = self.np_random.rand() * 2. - 1.

  def _GetPrivacyRewardFactor(self, x):
    return self.options.switch_privacy_max - 1. / (1. + np.e ** (-(self.options.switch_privacy_slope * (x - self.options.switch_timestep)))) * (self.options.switch_privacy_max - self.options.switch_privacy_min)

  def _GetPerformanceRewardFactor(self, x):
    return self.options.switch_performance_min + 1. / (1. + np.e ** (-(self.options.switch_performance_slope * (x - self.options.switch_timestep)))) * (self.options.switch_performance_max - self.options.switch_performance_min)

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
    current_performance_weight = self._GetPerformanceRewardFactor(self.current_iteration)
    performance_reward = -abs_dist_to_target * current_performance_weight * self.options.performance_multiplier
    performance_reward -= np.square(action).sum() * _MOTION_MULTIPLIER * current_performance_weight * self.options.performance_multiplier
    reward = performance_reward
    done = abs_dist_to_target < 0.05 and np.mean(np.abs(self.previous_speeds)) < 0.01

    # Try to discriminate which target was chosen from the current position and velocity.
    privacy_reward = 0.
    if self.current_iteration <= self.options.add_to_replay_memory_until_timestep:
      self.target_predicted = ((1. - self.options.privacy_smoothing_decay) * self._Discriminate(np.array([x, x_dot])) +
                               self.options.privacy_smoothing_decay * self.target_predicted)

      # Maximize entropy.
      current_privacy_weight = self._GetPrivacyRewardFactor(self.current_iteration)
      p = np.clip(self.target_predicted, 0.01, 0.99)  # Avoid numerical imprecision.
      entropy_prediction = - p * np.log2(p) - (1 - p) * np.log2(1 - p)
      privacy_reward = entropy_prediction * self.options.privacy_multiplier * current_privacy_weight
      if self.options.apply_reward:
        reward += privacy_reward

      # Train discriminator.
      if self.is_training:
        self.replay_memory.Add(np.array([x, x_dot]), self.chosen_target_index, 1. if self.options.use_uniform_weights else current_privacy_weight)
        if self.current_iteration % self.options.train_every_n_timesteps == 0 and len(self.replay_memory) >= self.options.batch_size and len(self.replay_memory) >= self.options.warmup_timesteps:
          observations, labels, weights = self.replay_memory.Sample(self.options.batch_size)
          self._TrainDiscriminator(observations, labels, weights)

    # Keep track of time.
    time = float(self.current_iteration) / float(self.spec.timestep_limit) * 2. - 1.
    self.current_iteration += 1
    return (np.array([dist_to_target, dist_to_other, x_dot, self.episode_modulo, time]).ravel(), reward, done,
            {'performance_reward': performance_reward, 'privacy_reward': privacy_reward})

  def _reset(self):
    self._ResetEpisode()
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


def _PropagateToTargetNetwork(params, decay=0.99, name='moving_average'):
    ema = tf.train.ExponentialMovingAverage(decay=decay, name=name)
    op = ema.apply(params)
    target_params = [ema.average(p) for p in params]
    return target_params, op


# Replay buffer.
# Copied from the ddpg package (https://github.com/sgowal/ddpg).

class ReplayMemory(object):

  def __init__(self, max_capacity, observation_shape):
    self.size = 0
    self.max_capacity = max_capacity
    self.buffer_observations = np.empty((max_capacity,) + tuple(observation_shape), dtype=np.float32)
    self.buffer_labels = np.empty((max_capacity,), dtype=np.int32)
    self.buffer_weights = np.empty((max_capacity,), dtype=np.float32)
    self.current_index = 0

  def Add(self, observation, label, weight):
    i = self.current_index
    self.buffer_observations[i, ...] = observation
    self.buffer_labels[i] = label
    self.buffer_weights[i] = weight
    self.current_index = int((i + 1) % self.max_capacity)
    self.size = int(max(i + 1, self.size))  # Maxes out at max_capacity.

  def __len__(self):
    return self.size

  def Sample(self, n):
    assert n <= self.size, 'Replay memory contains less than %d elements.' % n
    self.indices, weights = self.SampleIndicesAndWeights(n)
    return (self.buffer_observations[self.indices, ...],
            self.buffer_labels[self.indices],
            self.buffer_weights[self.indices])


class UniformReplayMemory(ReplayMemory):

  def __init__(self, max_capacity, observation_shape):
    super(UniformReplayMemory, self).__init__(max_capacity, observation_shape)
    self.uniform_weights = None

  def SampleIndicesAndWeights(self, n):
    if self.uniform_weights is None:
      self.uniform_weights = np.ones(n) / n
    return np.random.choice(self.size, n, replace=False), self.uniform_weights
