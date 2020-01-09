from gym.envs.registration import register

register(
    id='numberworld-v0',
    entry_point='gym_numberworld.envs:NumberWorldEnv',
)