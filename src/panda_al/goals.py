"""Goal types of the tasks, vectorized over environments (see :mod:`panda_al.taskspec` for their definitions).

Each goal gives the position the *pick object's center* has to reach (robot base frame, which is the env frame),
relative to the place object. The scripted expert plans towards it and the success term checks it. torch only.
"""

from __future__ import annotations

import torch

GOAL_IDS = {"stack": 0, "beside": 1}
HEIGHT_THRESHOLD = {"stack": 0.005, "beside": 0.01}
"""Allowed height error of the pick object's center [m]."""


def goal_target(
    goal_id: torch.Tensor,
    direction: torch.Tensor,
    distance: torch.Tensor,
    pick_size: torch.Tensor,
    place_size: torch.Tensor,
    place_pos: torch.Tensor,
) -> torch.Tensor:
    """Target position (n, 3) of the pick object's center."""
    stack = place_pos + torch.stack(
        [torch.zeros_like(distance), torch.zeros_like(distance), (pick_size[:, 2] + place_size[:, 2]) / 2], dim=-1
    )
    beside = torch.stack(
        [
            place_pos[:, 0],
            place_pos[:, 1] + direction * distance,
            place_pos[:, 2] - place_size[:, 2] / 2 + pick_size[:, 2] / 2,
        ],
        dim=-1,
    )
    return torch.where((goal_id == GOAL_IDS["stack"]).unsqueeze(-1), stack, beside)


def goal_reached(
    goal_id: torch.Tensor, pick_pos: torch.Tensor, target: torch.Tensor, xy_threshold: float = 0.04
) -> torch.Tensor:
    """Whether the pick object is at its target (position only; resting and gripper checks are the caller's)."""
    xy_ok = torch.linalg.norm(pick_pos[:, :2] - target[:, :2], dim=-1) < xy_threshold
    height_threshold = torch.where(
        goal_id == GOAL_IDS["stack"],
        torch.full_like(pick_pos[:, 2], HEIGHT_THRESHOLD["stack"]),
        torch.full_like(pick_pos[:, 2], HEIGHT_THRESHOLD["beside"]),
    )
    return xy_ok & ((pick_pos[:, 2] - target[:, 2]).abs() < height_threshold)
