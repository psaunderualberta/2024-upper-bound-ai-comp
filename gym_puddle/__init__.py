from gymnasium.envs.registration import register

register(
    id="PuddleWorld-v0",
    entry_point="gym_puddle.env.puddle_env:PuddleEnv",
)

register(
    id="PuddleWorldStochastic-v0",
    entry_point="gym_puddle.env.puddle_env_stochastic:PuddleEnvStochastic",
)

register(
    id="NoPuddleWorldStochastic-v0",
    entry_point="gym_puddle.env.no_puddle_env_stochastic:NoPuddleEnvStochastic"
)
