"""Task registrations, loaded by Isaac Lab through the ``isaaclab.tasks`` entry point."""

import gymnasium as gym

gym.register(
    id="PandaAL-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:PandaPickPlaceEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="PandaAL-MultiTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.multitask_env_cfg:PandaMultiTaskEnvCfg"},
    disable_env_checker=True,
)
