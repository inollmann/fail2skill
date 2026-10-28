"""Automatic failure tags, computed from a finished episode's recorded arrays (ground truth for the Cosmos analyst).

``stalled_at_start``  the arm hardly moved (TCP path shorter than 5 cm)
``no_grasp``          the arm moved but the pick object was never lifted (3 cm above its start height)
``dropped``           the object was lifted, but ended resting on the table away from the goal position
``misplaced``         the object was lifted and the goal was still not reached, without it lying on the table far away
                      (wrong spot or height, or still held at the end)
``collision``         the robot or the carried object touched the obstacle
``knocked_off``       the pick or place object fell off the table
``timeout``           the task's time limit ran out
``other``             none of the above

No simulator or Isaac Lab needed (torch only, for the goal definitions).
"""

from __future__ import annotations

import numpy as np
import torch

from .. import goals
from ..taskspec import BESIDE_DIRECTIONS

STALLED_PATH = 0.05
"""TCP path length [m] below which the robot counts as not having moved."""
LIFT_HEIGHT = 0.03
"""Height gain [m] of the pick object that counts as lifted."""
REST_HEIGHT = 0.01
"""An object within this height [m] of its resting height lies on the table."""
GOAL_XY = 0.04
"""Distance [m] to the goal position below which the object counts as at the goal (as in the success term)."""

TAGS = ("stalled_at_start", "no_grasp", "dropped", "misplaced", "collision", "knocked_off", "timeout", "other")


def failure_tags(data: dict[str, np.ndarray], object_names: list[str], task: dict, terms: list[str]) -> list[str]:
    """Tags of a failed episode.

    Args:
        data: per-step arrays of the episode (``cartesian_position``, ``object_pose`` (T, S, 7), ...).
        object_names: scene entity names of the ``object_pose`` columns (``obj_<name>``).
        task: ``extra.task`` of the episode (``pick``, ``place``, ``goal``).
        terms: termination terms that fired.
    """
    tags: list[str] = []
    tcp = data["cartesian_position"][:, :3]
    path = float(np.linalg.norm(np.diff(tcp, axis=0), axis=-1).sum()) if len(tcp) > 1 else 0.0
    stalled = path < STALLED_PATH
    if stalled:
        tags.append("stalled_at_start")

    poses = data["object_pose"]
    pick = poses[:, object_names.index(f"obj_{task['pick']}"), :3]
    place = poses[:, object_names.index(f"obj_{task['place']}"), :3]
    lifted = bool(pick[:, 2].max() - pick[0, 2] > LIFT_HEIGHT)
    if not stalled and not lifted:
        tags.append("no_grasp")
    if lifted:
        goal = task["goal"]
        target = goals.goal_target(
            torch.tensor([goals.GOAL_IDS[goal["type"]]]),
            torch.tensor([BESIDE_DIRECTIONS.get(goal.get("direction"), 1.0)]),
            torch.tensor([float(goal.get("distance", 0.0))]),
            torch.zeros(1, 3),  # sizes cancel out below: only the xy error and the resting height are used
            torch.zeros(1, 3),
            torch.as_tensor(place[-1:], dtype=torch.float32),
        ).numpy()[0]
        xy_error = float(np.linalg.norm(pick[-1, :2] - target[:2]))
        resting = bool(pick[-1, 2] - pick[0, 2] < REST_HEIGHT)
        tags.append("dropped" if resting and xy_error >= GOAL_XY else "misplaced")
    if "obstacle_collision" in terms:
        tags.append("collision")
    if "object_dropped" in terms:
        tags.append("knocked_off")
    if "time_out" in terms:
        tags.append("timeout")
    return tags or ["other"]
