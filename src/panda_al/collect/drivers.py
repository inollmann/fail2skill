"""What controls the robots during collection: a served policy, or the scripted expert (dry runs, baselines).

Both implement ``start(env_ids) -> {env_id: extra}`` (called after the envs were reset; ``extra`` goes into each
episode's ``meta.json``) and ``actions() -> (num_envs, 8)``. They need a running simulator.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dashcam import PANDA_FINGER_OPEN
from ..policy import ADAPTERS, BatchedPolicyRunner, Observation, PolicyRunner, PolicyUnavailable
from ..scripted import OBSTACLE_MODE, Corruption, ScriptedPickPlace, sample_corruption
from ..tasks.pick_place_env_cfg import PICK_OBJECT, PLACE_OBJECT
from .campaign import Campaign


def read_observations(
    env,
    camera_names: list[str],
    arm_ids: list[int],
    finger_id: int,
    instruction: str,
    image_envs: list[int] | None = None,
):
    """One :class:`Observation` per env from the current sim state (a multi-task env's prompts win over ``instruction``).

    Images are copied from the GPU only for ``image_envs`` (default: all envs); the others get an empty image dict.
    """
    infos = getattr(env, "episode_info", None)
    rows = None if image_envs is None else torch.as_tensor(image_envs, dtype=torch.long, device=env.device)
    images = {}
    for name in camera_names:
        rgb = env.scene[name].data.output["rgb"].torch[..., :3]
        images[name] = (rgb if rows is None else rgb[rows]).to(torch.uint8).cpu().numpy()
    index = {env_id: k for k, env_id in enumerate(range(env.num_envs) if image_envs is None else image_envs)}
    joint_pos = env.scene["robot"].data.joint_pos.torch
    closure = 1.0 - (joint_pos[:, finger_id] / PANDA_FINGER_OPEN).clamp(0.0, 1.0)
    arm = joint_pos[:, arm_ids].cpu().numpy().astype(np.float64)
    closure = closure.cpu().numpy().astype(np.float64)
    return [
        Observation(
            images={name: img[index[i]] for name, img in images.items()} if i in index else {},
            joint_position=arm[i],
            gripper_position=closure[i : i + 1],
            instruction=infos[i]["prompt"] if infos and infos[i] else instruction,
        )
        for i in range(env.num_envs)
    ]


class PolicyDriver:
    """Policy server control of all envs (see :mod:`panda_al.policy`).

    Batched (``policy.batch``): one connection, and every step one request for the envs that need a new action chunk.
    Otherwise one connection per env, one request per env and chunk.
    """

    def __init__(self, env, campaign: Campaign, instruction: str = ""):
        settings = campaign.policy
        self.env, self.settings, self.instruction = env, settings, instruction
        self.adapter = ADAPTERS[settings.adapter]()
        if settings.open_loop_horizon is not None:
            self.adapter.open_loop_horizon = settings.open_loop_horizon
        self.adapter.jpeg_quality = settings.jpeg_quality
        robot = env.scene["robot"]
        self.arm_ids = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        self.finger_id = robot.find_joints("panda_finger_joint1")[0][0]
        self.batched_runner: BatchedPolicyRunner | None = None
        self.runners: list[PolicyRunner] = []
        self.metadata: dict = {}
        if settings.batch != "false":
            runner = BatchedPolicyRunner(
                self.adapter, settings.host, settings.port, env.num_envs, settings.reconnect_attempts
            )
            try:
                self.metadata = runner.connect()
                self.batched_runner = runner
            except PolicyUnavailable:
                if settings.batch == "true" or runner.client is None:
                    raise  # required, or the server is not there at all
                runner.close()  # a server without batching: one connection per env
        if self.batched_runner is None:
            self.runners = [
                PolicyRunner(self.adapter, settings.host, settings.port, settings.reconnect_attempts)
                for _ in range(env.num_envs)
            ]

    @property
    def mode(self) -> str:
        return "batched" if self.batched_runner is not None else "per-env connections"

    def start(self, env_ids: list[int]) -> dict[int, dict]:
        extras = {}
        if self.batched_runner is not None:
            self.batched_runner.start_episodes(env_ids)
        for env_id in env_ids:
            metadata = self.metadata if self.batched_runner else self.runners[env_id].start_episode()
            extras[env_id] = {
                "policy": {
                    "label": self.settings.label,
                    "server": f"ws://{self.settings.host}:{self.settings.port}",
                    "batched": self.batched_runner is not None,
                    "metadata": metadata,
                    **self.adapter.describe(),
                }
            }
        return extras

    def actions(self) -> np.ndarray:
        cameras = self.adapter.camera_names()
        if self.batched_runner is None:
            observations = read_observations(self.env, cameras, self.arm_ids, self.finger_id, self.instruction)
            return np.stack([runner.act(obs) for runner, obs in zip(self.runners, observations, strict=True)])
        # images only for the envs that need a new chunk; every env still needs its joint state for the action
        need = self.batched_runner.needs_query()
        observations = read_observations(
            self.env, cameras, self.arm_ids, self.finger_id, self.instruction, image_envs=need
        )
        return np.stack(self.batched_runner.act(observations))

    def stats(self) -> str:
        runner = self.batched_runner
        if runner is None or runner.requests == 0:
            return ""
        return (
            f"policy: {runner.requests} requests, {runner.items / runner.requests:.1f} envs per request, "
            f"{runner.server_ms / runner.requests:.0f} ms model time per request"
        )

    def close(self):
        if self.batched_runner is not None:
            self.batched_runner.close()
        for runner in self.runners:
            runner.close()


class ExpertDriver:
    """The scripted expert, optionally corrupted: a no-VLA stand-in with a known mix of successes and failures."""

    def __init__(self, env, campaign: Campaign, generator: torch.Generator):
        self.env, self.settings, self.generator = env, campaign.expert, generator
        self.expert = ScriptedPickPlace(env, PICK_OBJECT, PLACE_OBJECT)
        self.modes = [m for m in self.settings.modes if m != OBSTACLE_MODE]  # the obstacle is a campaign perturbation

    def start(self, env_ids: list[int]) -> dict[int, dict]:
        corruptions = []
        for _ in env_ids:
            clean = torch.rand(1, generator=self.generator).item() >= self.settings.corrupt_ratio
            corruptions.append(Corruption() if clean or not self.modes else sample_corruption(self.modes, self.generator))
        self.expert.reset(env_ids, corruptions)
        return {
            env_id: {"generator": "scripted", "corruption": c.to_dict()} for env_id, c in zip(env_ids, corruptions)
        }

    def actions(self) -> torch.Tensor:
        return self.expert.compute()

    def stats(self) -> str:
        return ""

    def close(self):
        pass

