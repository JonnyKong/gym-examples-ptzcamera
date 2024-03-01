from gymnasium.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="gym_examples/PtzCamera",
    entry_point="gym_examples.envs:PtzCameraEnv",
)

register(
    id="gym_examples/UnrestrictedPtzCamera",
    entry_point="gym_examples.envs:UnrestrictedPtzCameraEnv",
)

register(
    id="gym_examples/PtzCameraReal",
    entry_point="gym_examples.envs:PtzCameraRealEnv",
)

register(
    id="gym_examples/UnrestrictedPtzCameraReal",
    entry_point="gym_examples.envs:UnrestrictedPtzCameraRealEnv",
)
