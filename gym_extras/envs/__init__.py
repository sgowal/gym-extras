import gym

from gym_extras.envs.continuous_cartpole import ContinuousCartPoleEnv
from gym_extras.envs.cart_reacher import CartReacherEnv
from gym_extras.envs.private_cart import PrivateCartEnv

try:
  from gym_extras.envs.cart import CartEnv
  from gym_extras.envs.choice_cart import ChoiceCartEnv
except gym.error.DependencyNotInstalled:
  pass
