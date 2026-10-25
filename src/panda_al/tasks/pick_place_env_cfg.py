"""Franka Panda pick-and-place task: put the red cube on top of the blue cube.

Built on Isaac Lab's maintained cube-stack scene (``isaaclab_tasks.contrib.stack``) with:

* **DROID-like action space** - absolute joint position targets for the 7 arm joints plus one gripper
  command (``>= 0.5`` closes the hand). This is the ``joint_pos`` 8-D action space that the Cosmos 3
  DROID post-training recipe uses, and it is exactly what a MoveIt trajectory or an IK solver produces.
* **15 Hz control** (DROID's control rate): physics at 120 Hz, decimation 8.
* **Three cameras at 640x360**: two exterior views on either side of the table and one wrist camera.
* **Dashcam** recorder term writing every episode to disk (see :mod:`panda_al.dashcam`).
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg
from isaaclab_tasks.contrib.stack.mdp import stack_events
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from ..dashcam import DashcamRecorderManagerCfg

TASK_NAME = "PandaAL-PickPlace-v0"
INSTRUCTION = "Pick up the red cube and place it on top of the blue cube."

PICK_OBJECT = "cube_2"
"""Red cube, the object to grasp."""
PLACE_OBJECT = "cube_1"
"""Blue cube, the object to place on."""

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
"""Cosmos 3's DROID recipe trains on 640x360 video."""

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
"""Custom sim assets (``assets/`` in the repo, mounted next to ``src/`` in the container)."""

TABLE_USD = ASSETS_DIR / "panda_table.usd"
TABLE_TOP_Z = 0.78
"""Height of the table top above the table origin [m]."""
TABLE_POS = (0.6, -0.8, -TABLE_TOP_Z)
"""The table origin is a corner; the table extends 0.76 m in -x and 1.6 m in +y. It spans x in [-0.16, 0.60],
y in [-0.8, 0.8] around the robot base, with the top at z = 0 (placed by hand in Isaac Lab)."""

ZED_MOUNT_USD = ASSETS_DIR / "panda_hand_zed_mount.usda"
MOUNT_POS = (0.001, 0.0, 0.007)
MOUNT_ROT = (0.0, 0.0, 1.0, 0.0)
"""Pose (x, y, z, w quaternion) of the mount CAD frame in the ``panda_hand`` frame (placed by hand in Isaac Lab:
translate 0.001, 0, 0.007, rotate 180 deg about z)."""

RTX_SETTINGS = {
    "antialiasing_mode": "DLAA",
    "enable_dl_denoiser": True,
    "samples_per_pixel": 4,
    "dome_light_upper_lower_strategy": 4,
}
"""Renderer quality for the recorded videos (the defaults are noisy at 640x360)."""

HOME_JOINT_POS = {
    # MoveIt "ready" state of franka_moveit_config, also multipanda's default initial position
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4,
    "panda_joint3": 0.0,
    "panda_joint4": -3 * math.pi / 4,
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2,
    "panda_joint7": math.pi / 4,
    "panda_finger_joint.*": 0.04,
}


def look_at_quat(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, ...]:
    """Quaternion (x, y, z, w) of a camera at ``eye`` looking at ``target`` in the ``"world"`` convention."""
    f = [t - e for t, e in zip(target, eye, strict=True)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    # y = up x f, z = f x y (up = +Z)
    y = [-f[1], f[0], 0.0]
    n = math.sqrt(sum(c * c for c in y))
    y = [c / n for c in y]
    z = [f[1] * y[2] - f[2] * y[1], f[2] * y[0] - f[0] * y[2], f[0] * y[1] - f[1] * y[0]]
    # rotation matrix with columns (f, y, z)
    m00, m01, m02 = f[0], y[0], z[0]
    m10, m11, m12 = f[1], y[1], z[1]
    m20, m21, m22 = f[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, yq, zq = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w, x, yq, zq = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w, x, yq, zq = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w, x, yq, zq = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return (x, yq, zq, w)


def exterior_camera(name: str, eye: tuple[float, float, float], target=(0.5, 0.0, 0.05)) -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=0.0,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, target), convention="world"),
    )


@configclass
class ObservationsCfg:
    """State observations (images are recorded by the dashcam directly from the camera sensors)."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        pick_object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(PICK_OBJECT)})
        pick_object_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(PICK_OBJECT)})
        place_object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(PLACE_OBJECT)})
        place_object_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(PLACE_OBJECT)})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pick_object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(PICK_OBJECT)}
    )
    place_object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(PLACE_OBJECT)}
    )
    # red on blue, gripper released and cubes at rest
    success = DoneTerm(
        func=mdp.cubes_stacked,
        params={
            "cube_1_cfg": SceneEntityCfg(PLACE_OBJECT),
            "cube_2_cfg": SceneEntityCfg(PICK_OBJECT),
            "cube_3_cfg": None,
        },
    )


@configclass
class EventCfg:
    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_cube_positions = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.35, 0.55), "y": (-0.2, 0.2), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.12,
            "asset_cfgs": [SceneEntityCfg(PLACE_OBJECT), SceneEntityCfg(PICK_OBJECT)],
        },
    )


@configclass
class PandaPickPlaceEnvCfg(FrankaCubeStackEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    recorders: DashcamRecorderManagerCfg = DashcamRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()

        # 15 Hz control, render once per control step
        self.sim.dt = 1.0 / 120.0
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.episode_length_s = 25.0

        # scene: two cubes only
        self.scene.cube_3 = None
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS, rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(TABLE_USD),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                semantic_tags=[("class", "table")],
            ),
        )
        self.scene.robot.init_state.joint_pos = dict(HOME_JOINT_POS)
        # stiff joint tracking without gravity sag (same as Isaac Lab's IK stack variants)
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.robot.actuators["panda_arm"].stiffness = 400.0
        self.scene.robot.actuators["panda_arm"].damping = 80.0

        self.events = EventCfg()

        # actions: [7 absolute arm joint targets, gripper command (0 open .. 1 closed)]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        # finger_joint2 is a passive mimic joint in the Isaac Lab Franka asset
        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint1"],
            open_command_expr={"panda_finger_joint1": 0.04},
            close_command_expr={"panda_finger_joint1": 0.0},
            threshold=0.5,
            positive_threshold=False,
        )

        # cameras (positions in the robot base / env frame; table top is at z = 0)
        self.scene.exterior_1 = exterior_camera("exterior_1", eye=(0.55, 0.85, 0.45))
        self.scene.exterior_2 = exterior_camera("exterior_2", eye=(0.55, -0.85, 0.45))
        # ZED mini on the custom hand mount (built by scripts/build_assets.py). The cameras are the
        # USD cameras of the asset, so intrinsics and lens poses live there.
        self.scene.zed_mount = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(Geometry/.*/)?panda_hand/zed_mount",
            init_state=AssetBaseCfg.InitialStateCfg(pos=MOUNT_POS, rot=MOUNT_ROT),
            spawn=sim_utils.UsdFileCfg(usd_path=str(ZED_MOUNT_USD), semantic_tags=[("class", "robot")]),
        )
        zed_path = "{ENV_REGEX_NS}/Robot/(Geometry/.*/)?panda_hand/zed_mount/zed_mini"
        self.scene.wrist = CameraCfg(
            prim_path=f"{zed_path}/left_cam",
            update_period=0.0,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
        )
        self.scene.wrist_right = CameraCfg(
            prim_path=f"{zed_path}/right_cam",
            update_period=0.0,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            data_types=["rgb"],
            spawn=None,
        )
        self.camera_names = ["exterior_1", "exterior_2", "wrist", "wrist_right"]
        self.depth_camera_names = ["wrist"]
        # DLAA and the DL denoiser accumulate over frames: re-render after resets so the first frame of an
        # episode does not show the previous episode
        self.num_rerenders_on_reset = 8
        for name in self.camera_names:
            set_isaac_rtx_global_settings(getattr(self.scene, name).renderer_cfg, **RTX_SETTINGS)

        # dashcam
        self.recorders.dashcam.task_name = TASK_NAME
        self.recorders.dashcam.instruction = INSTRUCTION
        self.recorders.dashcam.camera_names = list(self.camera_names)
        self.recorders.dashcam.depth_camera_names = list(self.depth_camera_names)
        self.recorders.dashcam.object_names = [PICK_OBJECT, PLACE_OBJECT]
