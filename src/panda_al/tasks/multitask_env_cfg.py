"""Multi-task variant of the pick-and-place task: ``PandaAL-MultiTask-v0``.

Same robot, cameras, dashcam, obstacle and 15 Hz control as :mod:`pick_place_env_cfg`, but the scene holds every
object of the registry (``assets/objects``) and each env draws its own task, objects, poses and prompt at every
reset from the task registry (``configs/tasks``). See :mod:`panda_al.tasks.multitask` and
``docs/automated_collection.md``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import illegal_contact
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.mdp import stack_events

from ..objects import ObjectSpec, load_objects
from . import multitask
from .pick_place_env_cfg import PandaPickPlaceEnvCfg

TASK_NAME = "PandaAL-MultiTask-v0"
MAX_EPISODE_S = 25.0
"""Upper bound of every task's ``time_limit_s``."""


def object_cfg(obj: ObjectSpec, index: int, count: int) -> RigidObjectCfg:
    """Scene entity of a registry object, initially parked under the table."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{obj.entity}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=multitask.park_position(index, count, obj)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj.resolved_usd_path(),
            scale=obj.scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass) if obj.mass is not None else None,
            physics_material=(
                sim_utils.RigidBodyMaterialCfg(
                    static_friction=obj.static_friction if obj.static_friction is not None else 0.5,
                    dynamic_friction=obj.dynamic_friction if obj.dynamic_friction is not None else 0.5,
                )
                if obj.static_friction is not None or obj.dynamic_friction is not None
                else None
            ),
            semantic_tags=[("class", obj.noun)],
        ),
    )


@configclass
class MultiTaskObservationsCfg:
    """State observations (images are recorded by the dashcam directly from the camera sensors)."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        object_poses = ObsTerm(func=multitask.pool_object_poses)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class MultiTaskTerminationsCfg:
    time_out = DoneTerm(func=multitask.task_time_out, time_out=True)
    object_dropped = DoneTerm(func=multitask.task_object_dropped)
    obstacle_collision = DoneTerm(
        func=illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("obstacle_contact")}
    )
    success = DoneTerm(func=multitask.task_success)


@configclass
class MultiTaskEventCfg:
    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_task_scene = EventTerm(
        func=multitask.ResetTaskScene,
        mode="reset",
        params={
            "pose_range": {"x": (0.35, 0.55), "y": (-0.2, 0.2), "yaw": (-1.0, 1.0)},
            "min_separation": 0.12,
        },
    )


@configclass
class PandaMultiTaskEnvCfg(PandaPickPlaceEnvCfg):
    observations: MultiTaskObservationsCfg = MultiTaskObservationsCfg()
    terminations: MultiTaskTerminationsCfg = MultiTaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = MAX_EPISODE_S

        # the registry replaces the two fixed cubes of the single-task scene
        self.scene.cube_1 = None
        self.scene.cube_2 = None
        objects = list(load_objects().values())
        for i, obj in enumerate(objects):
            setattr(self.scene, obj.entity, object_cfg(obj, i, len(objects)))

        self.events = MultiTaskEventCfg()
        self.recorders.dashcam.task_name = TASK_NAME
        self.recorders.dashcam.instruction = "multi-task"  # replaced by each episode's prompt
        self.recorders.dashcam.object_names = [obj.entity for obj in objects]
