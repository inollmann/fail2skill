"""Multi-task scene logic: per-env task, objects and prompt, and the terms that depend on them.

Everything is driven by the registries (:mod:`panda_al.objects`, :mod:`panda_al.taskspec`). Each registry object is a
scene entity ``obj_<name>`` in every env; objects that are not part of the current episode wait on the ground below
the table. :class:`ResetTaskScene` runs at every reset: it draws a task, its objects, their poses and a prompt, and
publishes the result as ``env.task_state`` (tensors, used by terms and the scripted expert) and
``env.episode_info`` (plain dicts, stored in each episode's ``meta.json`` by the dashcam).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.contrib.stack.mdp import stack_events

from .. import goals
from ..objects import ObjectSpec, load_objects
from ..scripted import OBSTACLE, OBSTACLE_PARK_POS
from ..taskspec import BESIDE_DIRECTIONS, TaskSpec, load_tasks

if TYPE_CHECKING:
    # heavy import (loads USD): the env cfg module is imported before the Kit app starts
    from isaaclab.envs import ManagerBasedRLEnv

GROUND_Z = -1.05
"""Height of the ground plane in the env frame (see the stack scene)."""


def park_position(index: int, count: int, obj: ObjectSpec) -> tuple[float, float, float]:
    """Waiting spot of the ``index``-th object: on the ground under the table, hidden by the table top."""
    return (0.40, (index - (count - 1) / 2) * 0.12, GROUND_Z + obj.rest_z)


class TaskState:
    """Per-env task description as tensors. Index 0..S-1 refers to the objects in registry order."""

    def __init__(self, env: ManagerBasedRLEnv, objects: list[ObjectSpec], tasks: list[TaskSpec]):
        n, d = env.num_envs, env.device
        self.objects = objects
        self.tasks = tasks
        self.keys = [obj.entity for obj in objects]
        self.sizes = torch.tensor([obj.size for obj in objects], device=d)
        self.task_idx = torch.zeros(n, dtype=torch.long, device=d)
        self.pick_idx = torch.zeros(n, dtype=torch.long, device=d)
        self.place_idx = torch.ones(n, dtype=torch.long, device=d)
        self.goal_id = torch.zeros(n, dtype=torch.long, device=d)
        self.goal_dir = torch.ones(n, device=d)
        self.goal_dist = torch.zeros(n, device=d)
        self.time_limit = torch.full((n,), float("inf"), device=d)

    def pool_pos_w(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """(n, S, 3) world positions of all objects."""
        return torch.stack([env.scene[k].data.root_pos_w.torch for k in self.keys], dim=1)

    def pool_quat_w(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.stack([env.scene[k].data.root_quat_w.torch for k in self.keys], dim=1)

    def pool_lin_vel_w(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.stack([env.scene[k].data.root_lin_vel_w.torch for k in self.keys], dim=1)

    def role_index(self, role: str) -> torch.Tensor:
        return self.pick_idx if role == "pick" else self.place_idx

    def role_pos_w(self, env: ManagerBasedRLEnv, role: str) -> torch.Tensor:
        pos = self.pool_pos_w(env)
        return pos[torch.arange(env.num_envs, device=env.device), self.role_index(role)]

    def role_quat_w(self, env: ManagerBasedRLEnv, role: str) -> torch.Tensor:
        quat = self.pool_quat_w(env)
        return quat[torch.arange(env.num_envs, device=env.device), self.role_index(role)]

    def target(self, place_pos: torch.Tensor) -> torch.Tensor:
        """Goal position of the pick object's center given the place object's position (env frame)."""
        return goals.goal_target(
            self.goal_id,
            self.goal_dir,
            self.goal_dist,
            self.sizes[self.pick_idx],
            self.sizes[self.place_idx],
            place_pos,
        )

    def set_env(self, env_id: int, task: TaskSpec, pick: ObjectSpec, place: ObjectSpec):
        names = [o.name for o in self.objects]
        goal = task.goal
        self.task_idx[env_id] = [t.name for t in self.tasks].index(task.name)
        self.pick_idx[env_id] = names.index(pick.name)
        self.place_idx[env_id] = names.index(place.name)
        self.goal_id[env_id] = goals.GOAL_IDS[goal["type"]]
        self.goal_dir[env_id] = BESIDE_DIRECTIONS.get(goal.get("direction"), 1.0)
        self.goal_dist[env_id] = float(goal.get("distance", 0.0))
        self.time_limit[env_id] = task.time_limit_s


class ResetTaskScene(ManagerTermBase):
    """Reset event: draw task, objects, poses and prompt for the resetting envs (see module docstring).

    Params (``EventTermCfg.params``): ``objects_dir``/``tasks_dir`` (registries; default repo directories),
    ``tasks`` (only these task names), ``task_weights`` (name -> weight overrides), ``pose_range`` and
    ``min_separation`` as in the stack scene's randomization.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = {k: v for k, v in cfg.params.items() if v is not None}  # the manager fills unset ones with None
        kwargs = {"root": params["objects_dir"]} if "objects_dir" in params else {}
        self.objects = list(load_objects(**kwargs).values())
        by_name = {o.name: o for o in self.objects}
        task_kwargs = {"root": params["tasks_dir"]} if "tasks_dir" in params else {}
        task_map = load_tasks(by_name, only=params.get("tasks"), **task_kwargs)
        self.tasks = list(task_map.values())
        weights = params.get("task_weights", {})
        self.weights = [float(weights.get(t.name, t.weight)) for t in self.tasks]
        self.pose_range = params.get("pose_range", {"x": (0.35, 0.55), "y": (-0.2, 0.2), "yaw": (-1.0, 1.0)})
        self.min_separation = params.get("min_separation", 0.12)
        self.state = TaskState(env, self.objects, self.tasks)
        env.task_state = self.state
        env.episode_info = [None] * env.num_envs
        env.restore_episode_info = self.restore
        self.pairs = {t.name: t.pairs(by_name) for t in self.tasks}
        self.by_name = by_name
        self.park = [park_position(i, len(self.objects), o) for i, o in enumerate(self.objects)]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        objects_dir: str | None = None,
        tasks_dir: str | None = None,
        tasks: list[str] | None = None,
        task_weights: dict[str, float] | None = None,
        pose_range: dict | None = None,
        min_separation: float | None = None,
    ):
        # all parameters are read once in __init__; they are listed so the event manager accepts them
        if env_ids is None or isinstance(env_ids, slice):  # the first reset passes slice(None)
            ids = list(range(env.num_envs))[env_ids if env_ids is not None else slice(None)]
        else:
            ids = [int(i) for i in env_ids]
        poses = {i: [] for i in range(len(self.objects))}  # object -> [(env id, x, y, z, yaw)]
        for env_id in ids:
            task = random.choices(self.tasks, weights=self.weights)[0]
            pick, place = random.choice(self.pairs[task.name])
            others = [o for o in self.objects if o.name not in (pick.name, place.name) and "distractor" in o.roles]
            count = min(random.randint(*task.distractors), len(others))
            active = [pick, place, *random.sample(others, count)]
            sampled = stack_events.sample_object_poses(len(active), self.min_separation, self.pose_range)
            where = {obj.name: pose for obj, pose in zip(active, sampled, strict=True)}
            for i, obj in enumerate(self.objects):
                if obj.name in where:
                    x, y, _, _, _, yaw = where[obj.name]
                    poses[i].append((env_id, x, y, obj.rest_z, yaw))
                else:
                    poses[i].append((env_id, *self.park[i][:3], 0.0))
            prompt, variant = task.render_prompt(pick, place)
            self.state.set_env(env_id, task, pick, place)
            env.episode_info[env_id] = {
                "task": task.name,
                "prompt": prompt,
                "prompt_variant": variant,
                "goal": dict(task.goal),
                "pick": pick.name,
                "place": place.name,
                "objects": [o.name for o in active],
                "time_limit_s": task.time_limit_s,
            }
        self._write(env, ids, poses)

    def _write(self, env: ManagerBasedRLEnv, ids: list[int], poses: dict[int, list]):
        device = env.device
        env_ids = torch.tensor(ids, device=device)
        origins = env.scene.env_origins[env_ids]
        for i, obj in enumerate(self.objects):
            rows = torch.tensor([row[1:] for row in poses[i]], device=device)  # x, y, z, yaw
            zeros = torch.zeros_like(rows[:, 3])
            quat = quat_from_euler_xyz(zeros, zeros, rows[:, 3])
            asset = env.scene[obj.entity]
            asset.write_root_pose_to_sim_index(root_pose=torch.cat([rows[:, :3] + origins, quat], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=device), env_ids=env_ids)
        # an obstacle left over from a previous episode is removed; the scripted expert may place it again
        if OBSTACLE in env.scene.keys():
            pos = torch.tensor(OBSTACLE_PARK_POS, device=device) + origins
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(len(ids), 1)
            env.scene[OBSTACLE].write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    def restore(self, env_ids: Sequence[int], infos: Sequence[dict]):
        """Make envs continue with the tasks of recorded episodes (replay); object poses come from the scene state."""
        for env_id, info in zip(env_ids, infos, strict=True):
            task = next(t for t in self.tasks if t.name == info["task"])
            self.state.set_env(env_id, task, self.by_name[info["pick"]], self.by_name[info["place"]])
            # the recorded goal and limit win over the current YAML, so old episodes stay reproducible
            goal = info["goal"]
            self.state.goal_id[env_id] = goals.GOAL_IDS[goal["type"]]
            self.state.goal_dir[env_id] = BESIDE_DIRECTIONS.get(goal.get("direction"), 1.0)
            self.state.goal_dist[env_id] = float(goal.get("distance", 0.0))
            self.state.time_limit[env_id] = float(info.get("time_limit_s", task.time_limit_s))
            self._env.episode_info[env_id] = dict(info)


"""
Terms.
"""


def _env_frame(env: ManagerBasedRLEnv, pos_w: torch.Tensor) -> torch.Tensor:
    """World positions (n, 3) or (n, S, 3) in each env's own frame (the robot base frame)."""
    origins = env.scene.env_origins
    return pos_w - origins.reshape(origins.shape[0], *([1] * (pos_w.dim() - 2)), 3)


def task_success(
    env: ManagerBasedRLEnv, xy_threshold: float = 0.04, max_lin_vel: float = 0.05, atol: float = 1e-4, rtol: float = 1e-4
) -> torch.Tensor:
    """Pick object at the goal of the env's task, gripper open, pick and place object at rest."""
    state: TaskState = env.task_state
    pick_pos = _env_frame(env, state.role_pos_w(env, "pick"))
    place_pos = _env_frame(env, state.role_pos_w(env, "place"))
    done = goals.goal_reached(state.goal_id, pick_pos, state.target(place_pos), xy_threshold)

    robot = env.scene["robot"]
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
    for joint_id in gripper_ids:
        done &= torch.isclose(robot.data.joint_pos.torch[:, joint_id], open_val, atol=atol, rtol=rtol)

    vel = state.pool_lin_vel_w(env)
    rows = torch.arange(env.num_envs, device=env.device)
    for role in ("pick", "place"):
        done &= torch.linalg.norm(vel[rows, state.role_index(role)], dim=-1) < max_lin_vel
    return done


def task_object_dropped(env: ManagerBasedRLEnv, minimum_height: float = -0.05) -> torch.Tensor:
    """The pick or place object fell off the table."""
    state: TaskState = env.task_state
    z = _env_frame(env, state.pool_pos_w(env))[..., 2]
    rows = torch.arange(env.num_envs, device=env.device)
    return (z[rows, state.pick_idx] < minimum_height) | (z[rows, state.place_idx] < minimum_height)


def task_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The task's own time limit (the env's global episode length is the upper bound)."""
    return env.episode_length_buf * env.step_dt >= env.task_state.time_limit


def pool_object_poses(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n, S*7) pose (position in the env frame + quaternion xyzw) of every registry object."""
    state: TaskState = env.task_state
    pos = _env_frame(env, state.pool_pos_w(env))
    return torch.cat([pos, state.pool_quat_w(env)], dim=-1).flatten(1)
