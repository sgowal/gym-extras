from gym.envs.registration import register

register(
    id='Continuous-CartPole-v0',
    entry_point='gym_extras.envs:ContinuousCartPoleEnv',
    timestep_limit=200,
    reward_threshold=190.,
)

register(
    id='Continuous-CartPole-v1',
    entry_point='gym_extras.envs:ContinuousCartPoleEnv',
    timestep_limit=300,
    reward_threshold=250.,
    kwargs={'target_location': 1.5, 'reward_sigma': 1.},
)
