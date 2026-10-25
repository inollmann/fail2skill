"""Scripted pick-and-place expert with optional, recorded failure injection ("corruptions").

The expert reads the object poses from the simulator at the start of each episode, plans a
sequence of top-down TCP waypoints (pre-grasp, grasp, lift, transport, place, release, retreat),
interpolates them with a minimum-jerk profile and converts them to absolute joint position
targets with differential IK. The resulting 8-D actions have the same format as MoveIt teleop
actions, so both data sources are interchangeable downstream.

The plan is executed open loop, so a corruption (e.g. a grasp target 4 cm off) is not corrected
and usually makes the episode fail. Whether it actually failed is decided by the task's success
term, not assumed. The sampled corruption is stored in the episode's ``meta.json``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, subtract_frame_transforms

HAND_TO_TCP = 0.1034
"""Distance from ``panda_hand`` to the fingertip center along the hand z-axis [m] (matches the env's ee_frame)."""

CUBE_SIZE = 0.0468
SAFE_HEIGHT = 0.15
"""TCP height above the table for lifting and transport [m]."""

# (name, duration [s], gripper command)
PHASES: list[tuple[str, float, float]] = [
    ("settle", 0.4, 0.0),
    ("pre_grasp", 1.6, 0.0),
    ("descend", 1.0, 0.0),
    ("close", 0.8, 1.0),
    ("lift", 1.0, 1.0),
    ("transport", 1.8, 1.0),
    ("lower", 1.0, 1.0),
    ("release", 0.6, 0.0),
    ("retreat", 1.0, 0.0),
    ("done", math.inf, 0.0),
]
PHASE_INDEX = {name: i for i, (name, _, _) in enumerate(PHASES)}


@dataclass
class Corruption:
    """Perturbation of the expert plan. All-zero means a clean demonstration."""

    mode: str = "none"
    grasp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Offset of the grasp position, robot base frame [m]."""
    grasp_yaw_offset: float = 0.0
    """Offset of the grasp yaw [rad]."""
    place_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Offset of the place position, robot base frame [m]."""
    release_at: float = -1.0
    """Fraction of the transport phase at which the gripper opens (negative: never)."""
    joint_noise_std: float = 0.0
    """Std of Gaussian noise added to the arm joint targets every step [rad]."""
    speed_scale: float = 1.0
    """Scales all phase durations (< 1 is faster)."""

    def to_dict(self) -> dict:
        return asdict(self)


def _random_xy(gen: torch.Generator, lo: float, hi: float) -> list[float]:
    r = lo + (hi - lo) * torch.rand(1, generator=gen).item()
    a = 2 * math.pi * torch.rand(1, generator=gen).item()
    return [r * math.cos(a), r * math.sin(a), 0.0]


def _uniform(gen: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=gen).item()


CORRUPTION_MODES = {
    "miss_grasp": lambda g: Corruption(mode="miss_grasp", grasp_offset=_random_xy(g, 0.03, 0.06)),
    "high_grasp": lambda g: Corruption(mode="high_grasp", grasp_offset=[0.0, 0.0, _uniform(g, 0.035, 0.06)]),
    "yaw_misalign": lambda g: Corruption(
        mode="yaw_misalign", grasp_yaw_offset=_uniform(g, 0.5, 0.75) * (1 if torch.rand(1, generator=g) < 0.5 else -1)
    ),
    "drop_in_transit": lambda g: Corruption(mode="drop_in_transit", release_at=_uniform(g, 0.2, 0.8)),
    "misplace": lambda g: Corruption(mode="misplace", place_offset=_random_xy(g, 0.035, 0.07)),
    "jitter": lambda g: Corruption(mode="jitter", joint_noise_std=_uniform(g, 0.02, 0.05)),
    "rushed": lambda g: Corruption(mode="rushed", speed_scale=_uniform(g, 0.25, 0.4)),
}
"""Failure modes and the magnitude ranges they are sampled from. Extend freely."""


def sample_corruption(modes: list[str], gen: torch.Generator) -> Corruption:
    mode = modes[int(torch.randint(len(modes), (1,), generator=gen).item())]
    return Corruption() if mode == "none" else CORRUPTION_MODES[mode](gen)


def _min_jerk(tau: torch.Tensor) -> torch.Tensor:
    tau = tau.clamp(0.0, 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


def _nlerp(q0: torch.Tensor, q1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
    q = q0 + s.unsqueeze(-1) * (q1 - q0)
    return q / q.norm(dim=-1, keepdim=True)


class ScriptedPickPlace:
    """Vectorized open-loop expert for :class:`~panda_al.tasks.pick_place_env_cfg.PandaPickPlaceEnvCfg`."""

    def __init__(self, env, pick_object: str, place_object: str):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.dt = env.step_dt
        self.robot = env.scene["robot"]
        self.pick = env.scene[pick_object]
        self.place = env.scene[place_object]
        self.arm_ids, _ = self.robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        # fixed-base robot: the root body is not part of the Jacobian
        self.jacobi_body_idx = self.hand_id - 1 if self.robot.is_fixed_base else self.hand_id
        self.jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.arm_ids]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=self.device,
        )

        n, d = self.num_envs, self.device
        self.phase = torch.zeros(n, dtype=torch.long, device=d)
        self.phase_time = torch.zeros(n, device=d)
        self.planned = torch.zeros(n, dtype=torch.bool, device=d)
        self.start_pos = torch.zeros(n, 3, device=d)
        self.start_quat = torch.zeros(n, 4, device=d)
        self.waypoints = torch.zeros(n, len(PHASES), 3, device=d)
        self.grasp_quat = torch.zeros(n, 4, device=d)
        self.durations = torch.tensor([p[1] for p in PHASES], device=d).repeat(n, 1)
        self.gripper_by_phase = torch.tensor([p[2] for p in PHASES], device=d)
        # per-env corruption parameters
        self.corruptions: list[Corruption] = [Corruption() for _ in range(n)]
        self.grasp_offset = torch.zeros(n, 3, device=d)
        self.grasp_yaw_offset = torch.zeros(n, device=d)
        self.place_offset = torch.zeros(n, 3, device=d)
        self.release_at = -torch.ones(n, device=d)
        self.joint_noise_std = torch.zeros(n, device=d)

    def reset(self, env_ids: list[int], corruptions: list[Corruption]):
        """Start a new plan for ``env_ids`` (call after the env reset them)."""
        for env_id, corruption in zip(env_ids, corruptions, strict=True):
            self.corruptions[env_id] = corruption
            self.grasp_offset[env_id] = torch.tensor(corruption.grasp_offset, device=self.device)
            self.grasp_yaw_offset[env_id] = corruption.grasp_yaw_offset
            self.place_offset[env_id] = torch.tensor(corruption.place_offset, device=self.device)
            self.release_at[env_id] = corruption.release_at
            self.joint_noise_std[env_id] = corruption.joint_noise_std
            self.durations[env_id] = torch.tensor([p[1] for p in PHASES], device=self.device) * corruption.speed_scale
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.phase[ids] = 0
        self.phase_time[ids] = 0.0
        self.planned[ids] = False

    def _hand_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        root = self.robot.data.root_pose_w.torch
        hand = self.robot.data.body_pose_w.torch[:, self.hand_id]
        return subtract_frame_transforms(root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7])

    def _object_pose_b(self, obj) -> tuple[torch.Tensor, torch.Tensor]:
        root = self.robot.data.root_pose_w.torch
        return subtract_frame_transforms(
            root[:, :3], root[:, 3:7], obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
        )

    def _plan(self, ids: torch.Tensor, hand_pos: torch.Tensor, hand_quat: torch.Tensor):
        """Compute the waypoints of ``ids`` from the current object poses."""
        tcp_pos = hand_pos + quat_apply(hand_quat, torch.tensor([0.0, 0.0, HAND_TO_TCP], device=self.device))
        pick_pos, pick_quat = self._object_pose_b(self.pick)
        place_pos, _ = self._object_pose_b(self.place)
        # top-down grasp; pick the cube yaw (mod 90 deg) closest to the current hand yaw
        _, _, hand_yaw = euler_xyz_from_quat(hand_quat)
        _, _, cube_yaw = euler_xyz_from_quat(pick_quat)
        k = torch.round((hand_yaw - cube_yaw) / (math.pi / 2))
        yaw = cube_yaw + k * math.pi / 2 + self.grasp_yaw_offset
        roll = torch.full_like(yaw, math.pi)
        grasp_quat = quat_from_euler_xyz(roll, torch.zeros_like(yaw), yaw)

        grasp = pick_pos + self.grasp_offset
        place = place_pos + self.place_offset + torch.tensor([0.0, 0.0, CUBE_SIZE + 0.01], device=self.device)
        up = lambda p, z: torch.cat([p[:, :2], torch.full_like(p[:, 2:], z)], dim=-1)  # noqa: E731
        wp = torch.stack(
            [
                tcp_pos,  # settle
                grasp + torch.tensor([0.0, 0.0, 0.12], device=self.device),  # pre_grasp
                grasp,  # descend
                grasp,  # close
                up(grasp, SAFE_HEIGHT),  # lift
                up(place, SAFE_HEIGHT),  # transport
                place,  # lower
                place,  # release
                place + torch.tensor([0.0, 0.0, 0.10], device=self.device),  # retreat
                place + torch.tensor([0.0, 0.0, 0.10], device=self.device),  # done
            ],
            dim=1,
        )
        self.waypoints[ids] = wp[ids]
        self.grasp_quat[ids] = grasp_quat[ids]
        self.start_pos[ids] = tcp_pos[ids]
        self.start_quat[ids] = hand_quat[ids]
        self.planned[ids] = True

    def compute(self) -> torch.Tensor:
        """Return the (num_envs, 8) action for the current step and advance the phase clocks."""
        hand_pos, hand_quat = self._hand_pose_b()
        new = (~self.planned).nonzero(as_tuple=False).squeeze(-1)
        if len(new) > 0:
            self._plan(new, hand_pos, hand_quat)

        envs = torch.arange(self.num_envs, device=self.device)
        phase = self.phase
        target = self.waypoints[envs, phase]
        tau = _min_jerk(self.phase_time / self.durations[envs, phase])
        pos = self.start_pos + tau.unsqueeze(-1) * (target - self.start_pos)
        # rotate to the grasp orientation during the first two phases, then hold it
        quat_tau = torch.where(phase <= PHASE_INDEX["pre_grasp"], tau, torch.ones_like(tau))
        quat = _nlerp(self.start_quat, self.grasp_quat, quat_tau)
        quat = torch.where((phase == 0).unsqueeze(-1), self.start_quat, quat)

        # TCP target -> hand target -> joint targets
        hand_target = pos - quat_apply(quat, torch.tensor([0.0, 0.0, HAND_TO_TCP], device=self.device))
        self.ik.set_command(torch.cat([hand_target, quat], dim=-1))
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.jacobi_body_idx, :, self.jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_ids]
        joint_target = self.ik.compute(hand_pos, hand_quat, jacobian, joint_pos)
        joint_target = joint_target + torch.randn_like(joint_target) * self.joint_noise_std.unsqueeze(-1)

        gripper = self.gripper_by_phase[phase].clone()
        transport = phase == PHASE_INDEX["transport"]
        dropped = (self.release_at >= 0) & ((transport & (tau >= self.release_at)) | (phase > PHASE_INDEX["transport"]))
        gripper = torch.where(dropped, torch.zeros_like(gripper), gripper)

        # advance phase clocks; the end pose of a phase is the start pose of the next one
        self.phase_time += self.dt
        finished = self.phase_time >= self.durations[envs, phase]
        if finished.any():
            ids = finished.nonzero(as_tuple=False).squeeze(-1)
            self.start_pos[ids] = target[ids]
            self.start_quat[ids] = quat[ids]
            self.phase[ids] = (phase[ids] + 1).clamp(max=len(PHASES) - 1)
            self.phase_time[ids] = 0.0
        return torch.cat([joint_target, gripper.unsqueeze(-1)], dim=-1)
