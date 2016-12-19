import gym

from gym_extras.envs.continuous_cartpole import ContinuousCartPoleEnv
from gym_extras.envs.cart_reacher import CartReacherEnv

try:
  from gym_extras.envs.cart import CartEnv
  from gym_extras.envs.choice_cart import ChoiceCartEnv
except gym.error.DependencyNotInstalled:
  pass
