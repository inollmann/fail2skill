"""Dashcam: an Isaac Lab recorder term that writes every episode (successful or not) to disk.

Because it is a :class:`~isaaclab.managers.RecorderTerm`, the environment calls it from inside
``env.step()`` / ``env.reset()``. Every way of driving the robot (MoveIt teleop, scripted generator,
replay of Cosmos-generated actions) is therefore recorded identically, without extra code in
the driver scripts.

Per step it records the camera frames, robot/object state and the action about to be applied;
when an env resets it adds the terminal frame and files the episode under ``success/`` or
``failure/`` based on the task's ``success`` termination term. See :mod:`panda_al.episode_io`
for the format.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

from .episode_io import EpisodeMeta, EpisodeWriter, flatten_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PANDA_FINGER_OPEN = 0.04
"""Finger joint position [m] of a fully open Panda hand."""


class DashcamRecorder(RecorderTerm):
    """Records camera videos, actions and state of every episode. See module docstring."""

    cfg: DashcamRecorderCfg

    def __init__(self, cfg: DashcamRecorderCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._arm_ids, self._arm_names = self._robot.find_joints(cfg.arm_joint_names, preserve_order=True)
        self._finger_id = self._robot.find_joints(cfg.finger_joint_name)[0][0]
        self._ee_frame = env.scene[cfg.ee_frame_name]
        self._objects = {name: env.scene[name] for name in cfg.object_names}
        self._cameras = {name: env.scene[name] for name in cfg.camera_names}
        self._depth_cameras = {name: env.scene[name] for name in cfg.depth_camera_names}
        self._fps = 1.0 / env.step_dt
        self._run_dir = cfg.output_dir
        self._writers: dict[int, EpisodeWriter] = {}
        self._initial_states: dict[int, dict[str, np.ndarray]] = {}
        # the recorder manager is built before the action manager, so this is allocated on the first step
        self._last_action: torch.Tensor | None = None
        self._tag = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self._counter = 0

        self.enabled = cfg.enabled
        """Recording switch; driver scripts may toggle it (e.g. to skip warm-up)."""
        self.outcome_override: dict[int, str] = {}
        """Per env id: force the outcome (``success``/``failure``/``discard``) of the running episode."""
        self.episode_extra: dict[int, dict] = {}
        """Per env id: extra metadata stored in ``meta.json`` of the running episode (consumed on finalize)."""
        self.parent_episode: dict[int, str] = {}
        """Per env id: id of the episode the running one was derived from (replays)."""
        self.last_finished: dict[int, tuple[str, str | None]] = {}
        """Per env id: (outcome, path) of the most recently finished episode, for driver scripts to report."""

    """
    Recorder callbacks.
    """

    def record_post_reset(self, env_ids: Sequence[int] | None):
        self.snapshot_initial_state(env_ids)
        return None, None

    def snapshot_initial_state(self, env_ids: Sequence[int] | None = None):
        """Store the current scene state as the initial state of the next episode, so it can be replayed exactly.

        Called after every reset; drivers that change the scene right after the reset (e.g. the scripted
        generator placing an obstacle between the cubes) call it again.
        """
        state = self._env.scene.get_state(is_relative=True)
        for env_id in self._resolve(env_ids):
            self._initial_states[env_id] = flatten_state(_index_state(state, env_id))

    def record_pre_step(self):
        if not self.enabled:
            return None, None
        action = self._env.action_manager.action
        self._last_action = action.clone()
        self._append_rows(range(self._env.num_envs), action, is_last=False)
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        env_ids = [i for i in self._resolve(env_ids) if i in self._writers]
        if not env_ids:
            return None, None
        # terminal observation; the action column repeats the last applied action
        self._append_rows(env_ids, self._last_action, is_last=True)
        fired = self._fired_terminations(env_ids)
        for env_id in env_ids:
            writer = self._writers.pop(env_id)
            outcome = self.outcome_override.pop(env_id, None)
            if outcome is None:
                outcome = "success" if "success" in fired[env_id] else "failure"
            writer.meta.extra.update(self.episode_extra.pop(env_id, {}))
            writer.meta.parent_episode = self.parent_episode.pop(env_id, None)
            path = writer.finalize(outcome, fired[env_id])
            self.last_finished[env_id] = (outcome, str(path) if path else None)
            if path is not None:
                print(f"[dashcam] env {env_id}: {outcome:<8} {writer.num_rows:4d} steps -> {path}")
        return None, None

    def close(self, file_path: str | None = None):
        # episodes still running when the app shuts down are discarded
        for writer in self._writers.values():
            writer.finalize("discard")
        self._writers.clear()

    """
    Helpers.
    """

    def _resolve(self, env_ids) -> list[int]:
        if env_ids is None or isinstance(env_ids, slice):
            return list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.tolist()
        return list(env_ids)

    def _new_writer(self, env_id: int) -> EpisodeWriter:
        self._counter += 1
        episode_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{self.cfg.source}_{self._tag}_{self._counter:05d}"
        meta = EpisodeMeta(
            episode_id=episode_id,
            task=self.cfg.task_name,
            instruction=self.cfg.instruction,
            source=self.cfg.source,
            fps=self._fps,
            cameras=list(self.cfg.camera_names),
            depth_cameras=list(self.cfg.depth_camera_names),
            object_names=list(self.cfg.object_names),
            arm_joint_names=list(self._arm_names),
        )
        writer = EpisodeWriter(self._run_dir, meta)
        if env_id in self._initial_states:
            writer.set_initial_state(self._initial_states[env_id])
        return writer

    def _append_rows(self, env_ids, action: torch.Tensor, is_last: bool):
        env_ids = list(env_ids)
        frames = {
            name: cam.data.output["rgb"].torch[..., :3].to(torch.uint8).cpu().numpy()
            for name, cam in self._cameras.items()
        }
        depth = {name: self._depth_mm(cam) for name, cam in self._depth_cameras.items()}
        step = self._state_numpy()
        action_np = action.detach().float().cpu().numpy()
        for env_id in env_ids:
            writer = self._writers.get(env_id)
            if writer is None:
                writer = self._writers[env_id] = self._new_writer(env_id)
            row = {key: value[env_id] for key, value in step.items()}
            row["action"] = action_np[env_id]
            row["is_first"] = np.bool_(writer.num_rows == 0)
            row["is_last"] = np.bool_(is_last)
            writer.append(
                {name: f[env_id] for name, f in frames.items()}, row, {name: d[env_id] for name, d in depth.items()}
            )

    def _depth_mm(self, camera) -> np.ndarray:
        """Depth image (N, H, W) uint16 in mm; 0 where invalid or outside the ZED mini's depth range."""
        depth = camera.data.output["distance_to_image_plane"].torch[..., 0]
        valid = (depth >= self.cfg.depth_range[0]) & (depth <= self.cfg.depth_range[1])
        depth_mm = torch.where(valid, depth * 1000.0, torch.zeros_like(depth)).round()
        return depth_mm.to(torch.int32).cpu().numpy().astype(np.uint16)

    def _state_numpy(self) -> dict[str, np.ndarray]:
        data = self._robot.data
        joint_pos = data.joint_pos.torch
        ee_pos = self._ee_frame.data.target_pos_source.torch[:, 0]
        ee_quat = self._ee_frame.data.target_quat_source.torch[:, 0]
        roll, pitch, yaw = euler_xyz_from_quat(ee_quat)
        origins = self._env.scene.env_origins
        object_pose = torch.stack(
            [
                torch.cat([obj.data.root_pos_w.torch - origins, obj.data.root_quat_w.torch], dim=-1)
                for obj in self._objects.values()
            ],
            dim=1,
        )
        closure = 1.0 - (joint_pos[:, self._finger_id] / PANDA_FINGER_OPEN).clamp(0.0, 1.0)
        out = {
            "joint_position": joint_pos[:, self._arm_ids],
            "joint_velocity": data.joint_vel.torch[:, self._arm_ids],
            "gripper_position": closure.unsqueeze(-1),
            "cartesian_position": torch.cat([ee_pos, torch.stack([roll, pitch, yaw], dim=-1)], dim=-1),
            "ee_quat": ee_quat,
            "object_pose": object_pose,
        }
        out = {key: value.detach().float().cpu().numpy() for key, value in out.items()}
        out["timestamp"] = self._env.episode_length_buf.detach().cpu().numpy().astype(np.float64) * self._env.step_dt
        return out

    def _fired_terminations(self, env_ids: list[int]) -> dict[int, list[str]]:
        manager = self._env.termination_manager
        fired = {env_id: [] for env_id in env_ids}
        for name in manager.active_terms:
            values = manager.get_term(name)
            for env_id in env_ids:
                if bool(values[env_id]):
                    fired[env_id].append(name)
        return fired


def _index_state(state: dict, env_id: int) -> dict:
    """Select one environment from a nested ``scene.get_state()`` dict (keeping a batch dim of 1)."""
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            out[key] = _index_state(value, env_id)
        else:
            out[key] = value[env_id : env_id + 1].detach().cpu().numpy()
    return out


@configclass
class DashcamRecorderCfg(RecorderTermCfg):
    """Configuration of :class:`DashcamRecorder`."""

    class_type: type[RecorderTerm] = DashcamRecorder

    enabled: bool = True
    output_dir: str = os.environ.get("AL_DATA_DIR", "/workspace/data") + "/episodes/default"
    """Run directory; driver scripts usually override it."""
    source: str = "unspecified"
    task_name: str = MISSING
    instruction: str = MISSING
    camera_names: list[str] = MISSING
    object_names: list[str] = MISSING
    depth_camera_names: list[str] = []
    """Cameras whose ``distance_to_image_plane`` is stored as a 16-bit depth video (mm)."""
    depth_range: tuple[float, float] = (0.1, 15.0)
    """Valid depth [m] (ZED mini); everything else is stored as 0."""
    arm_joint_names: list[str] = ["panda_joint[1-7]"]
    finger_joint_name: str = "panda_finger_joint1"
    ee_frame_name: str = "ee_frame"


@configclass
class DashcamRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager with only the dashcam (no Isaac Lab HDF5 export)."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    dashcam: DashcamRecorderCfg = DashcamRecorderCfg()


def get_dashcam(env: ManagerBasedRLEnv) -> DashcamRecorder:
    """Return the env's dashcam term (the recorder manager has no public accessor for its terms)."""
    return env.recorder_manager._terms["dashcam"]
