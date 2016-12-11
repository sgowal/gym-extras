from gym.envs.registration import register

register(
    id='Continuous-CartPole-v0',
    entry_point='gym_private.envs:ContinuousCartPoleEnv',
    timestep_limit=200,
    reward_threshold=190.0,
)
