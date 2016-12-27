from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym_extras.mujoco.cart import CartEnv
from gym_extras.mujoco.choice_cart import ChoiceCartEnv
