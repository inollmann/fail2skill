"""Task registry: every ``configs/tasks/*.yaml`` defines a task (goal, object roles, prompt templates).

Pure Python (no Isaac Lab). The goal types themselves are implemented in :mod:`panda_al.goals`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .objects import ObjectSpec
from .paths import TASKS_DIR

GOAL_TYPES = ("stack", "beside")
BESIDE_DIRECTIONS = {"left": 1.0, "right": -1.0}
"""Direction -> sign of the y offset in the robot base frame (the robot looks along +x, its left is +y)."""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    goal: dict
    prompts: tuple[str, ...]
    pick: dict = field(default_factory=dict)
    place: dict = field(default_factory=dict)
    distractors: tuple[int, int] = (0, 1)
    time_limit_s: float = 25.0
    weight: float = 1.0

    def accepts(self, role: str, obj: ObjectSpec) -> bool:
        """Whether ``obj`` may take ``role`` (``pick`` or ``place``) in this task."""
        if role not in obj.roles:
            return False
        for key, wanted in (self.pick if role == "pick" else self.place).items():
            allowed = wanted if isinstance(wanted, list) else [wanted]
            if getattr(obj, key) not in allowed:
                return False
        return True

    def pairs(self, objects: dict[str, ObjectSpec]) -> list[tuple[ObjectSpec, ObjectSpec]]:
        """All valid (pick, place) object combinations."""
        return [
            (p, q)
            for p in objects.values()
            for q in objects.values()
            if p.name != q.name and self.accepts("pick", p) and self.accepts("place", q)
        ]

    def render_prompt(self, pick: ObjectSpec, place: ObjectSpec, rng: random.Random | None = None) -> tuple[str, int]:
        """A random phrasing of the task for these objects, and its index."""
        variant = (rng or random).randrange(len(self.prompts))
        return self.prompts[variant].format(pick=pick, place=place), variant


def _task_from_yaml(path: Path) -> TaskSpec:
    data = yaml.safe_load(path.read_text())
    try:
        goal = dict(data["goal"])
        if goal.get("type") not in GOAL_TYPES:
            raise ValueError(f"goal type must be one of {GOAL_TYPES}")
        if goal["type"] == "beside" and goal.get("direction") not in BESIDE_DIRECTIONS:
            raise ValueError(f"beside needs direction in {sorted(BESIDE_DIRECTIONS)}")
        prompts = tuple(str(p) for p in data["prompts"])
        if not prompts:
            raise ValueError("at least one prompt is required")
        low, high = (int(v) for v in data.get("distractors", (0, 1)))
        return TaskSpec(
            name=str(data.get("name", path.stem)),
            goal=goal,
            prompts=prompts,
            pick=dict(data.get("pick", {})),
            place=dict(data.get("place", {})),
            distractors=(low, high),
            time_limit_s=float(data.get("time_limit_s", 25.0)),
            weight=float(data.get("weight", 1.0)),
        )
    except (KeyError, ValueError, TypeError) as err:
        raise ValueError(f"{path}: invalid task definition: {err!r}") from err


def load_tasks(
    objects: dict[str, ObjectSpec], root: str | Path = TASKS_DIR, only: list[str] | None = None
) -> dict[str, TaskSpec]:
    """All tasks below ``root`` (or the ones named in ``only``), validated against the object registry."""
    tasks: dict[str, TaskSpec] = {}
    for path in sorted(Path(root).glob("*.yaml")):
        task = _task_from_yaml(path)
        if only is not None and task.name not in only:
            continue
        if task.name in tasks:
            raise ValueError(f"{path}: duplicate task name {task.name!r}")
        if not task.pairs(objects):
            raise ValueError(f"{path}: no pair of objects fits the pick/place filters of task {task.name!r}")
        for prompt in task.prompts:  # catch typos in templates now, not mid-run
            first, second = task.pairs(objects)[0]
            prompt.format(pick=first, place=second)
        tasks[task.name] = task
    missing = set(only or []) - set(tasks)
    if missing:
        raise ValueError(f"unknown tasks {sorted(missing)} (available in {root}: see *.yaml)")
    if not tasks:
        raise ValueError(f"no tasks found in {root}")
    return tasks
