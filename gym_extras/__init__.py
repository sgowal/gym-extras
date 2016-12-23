import gym
from gym.envs.registration import register

register(
    id='Continuous-CartPole-v0',
    entry_point='gym_extras.envs:ContinuousCartPoleEnv',
    timestep_limit=200,
    reward_threshold=1950.,
)

register(
    id='Continuous-CartPole-v1',
    entry_point='gym_extras.envs:ContinuousCartPoleEnv',
    timestep_limit=300,
    reward_threshold=2800.,
    kwargs={'target_location': 1.5},
)

register(
    id='Cart-Reacher-v0',
    entry_point='gym_extras.envs:CartReacherEnv',
    timestep_limit=50,
    reward_threshold=-7.,
)

register(
    id='Private-Cart-v0',
    entry_point='gym_extras.envs:PrivateCartEnv',
    timestep_limit=80,
)

register(
    id='Cart-v0',
    entry_point='gym_extras.envs:CartEnv',
    timestep_limit=50,
    reward_threshold=-6.,
    kwargs={'target_location': 0.0},
)

register(
    id='Choice-Cart-v0',
    entry_point='gym_extras.envs:ChoiceCartEnv',
    timestep_limit=50,
    reward_threshold=-14.,
)
