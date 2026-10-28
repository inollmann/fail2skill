"""Placing the cylinder obstacle between the pick and the place object (shared by the scripted expert and collector).

The obstacle is a kinematic cylinder scene entity (``obstacle``) that waits under the table. For an episode with an
``obstacle_cylinder`` corruption it is moved between the cubes, right after the env reset has assigned their poses.
Whoever calls :func:`place_obstacles` has to call ``dashcam.snapshot_initial_state`` afterwards so that replays contain
the obstacle too.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from .scripted import OBSTACLE, OBSTACLE_MODE, OBSTACLE_PARK_POS, Corruption

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

RolePose = Callable[[str], tuple[torch.Tensor, torch.Tensor]]
"""``role ("pick" | "place") -> (position, quaternion)`` of that object for all envs, robot base frame."""


def task_state_role_pose(env: ManagerBasedRLEnv, role: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of the pick / place object of each env's current task (multi-task env), robot base frame."""
    state = env.task_state
    root = env.scene["robot"].data.root_pose_w.torch
    return subtract_frame_transforms(
        root[:, :3], root[:, 3:7], state.role_pos_w(env, role), state.role_quat_w(env, role)
    )


def place_obstacles(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], corruptions: Sequence[Corruption], role_pose: RolePose
) -> None:
    """Stand the obstacle between the cubes in envs whose corruption is ``obstacle_cylinder``, park it elsewhere.

    Fills ``corruption.obstacle_position`` (robot base frame). Re-renders when the obstacle moved, because the cameras
    were rendered at the reset, before it appeared.
    """
    obstacle = env.scene[OBSTACLE] if OBSTACLE in env.scene.keys() else None
    if obstacle is None:
        return
    device = env.device
    pick, _ = role_pose("pick")
    place, _ = role_pose("place")
    root = env.scene["robot"].data.root_pose_w.torch
    positions = []
    for env_id, corruption in zip(env_ids, corruptions, strict=True):
        if corruption.mode == OBSTACLE_MODE:
            line = (place[env_id] - pick[env_id])[:2]
            perp = torch.stack([-line[1], line[0]]) / line.norm()
            xy = pick[env_id, :2] + corruption.obstacle_along * line + corruption.obstacle_lateral * perp
            pos_b = torch.cat([xy, xy.new_tensor([corruption.obstacle_height / 2])])
            corruption.obstacle_position = [round(v, 5) for v in pos_b.tolist()]
        else:
            pos_b = torch.tensor(OBSTACLE_PARK_POS, device=device)
        positions.append(root[env_id, :3] + quat_apply(root[env_id, 3:7], pos_b))
    ids = torch.as_tensor(list(env_ids), device=device, dtype=torch.long)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(len(ids), 1)
    obstacle.write_root_pose_to_sim_index(root_pose=torch.cat([torch.stack(positions), quat], dim=-1), env_ids=ids)

    active = getattr(env, "_obstacle_active", None)
    if active is None:
        active = env._obstacle_active = [False] * env.num_envs
    used = [c.mode == OBSTACLE_MODE for c in corruptions]
    moved = any(used) or any(active[i] for i in env_ids)
    for env_id, is_used in zip(env_ids, used, strict=True):
        active[env_id] = is_used
    if moved:
        env.sim.forward()
        for _ in range(env.cfg.num_rerenders_on_reset):
            env.sim.render()
