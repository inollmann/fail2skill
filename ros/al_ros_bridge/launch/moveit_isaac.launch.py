"""MoveIt 2 + RViz for the Isaac Lab Panda, using multipanda_ros2's description and MoveIt config.

Same move_group setup as ``franka_moveit_config/launch/sim_moveit.launch.py``, but the robot is
the Isaac Lab simulation behind ``al_ros_bridge`` instead of MuJoCo + ros2_control. All nodes
run on simulation time published by the bridge.

    ros2 launch al_ros_bridge moveit_isaac.launch.py [rviz:=false] [sim_host:=127.0.0.1] [sim_port:=45210]
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package: str, path: str):
    with open(os.path.join(get_package_share_directory(package), path)) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    sim_time = {"use_sim_time": True}
    description_dir = get_package_share_directory("franka_description")
    moveit_dir = get_package_share_directory("franka_moveit_config")

    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    os.path.join(description_dir, "robots", "real", "panda_arm.urdf.xacro"),
                    " hand:=true use_fake_hardware:=true",
                ]
            ),
            value_type=str,
        )
    }
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    os.path.join(moveit_dir, "srdf", "panda_arm.srdf.xacro"),
                    " hand:=true",
                ]
            ),
            value_type=str,
        )
    }
    kinematics = load_yaml("franka_moveit_config", "config/kinematics.yaml")

    ompl = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": "default_planner_request_adapters/AddTimeOptimalParameterization "
            "default_planner_request_adapters/ResolveConstraintFrames "
            "default_planner_request_adapters/FixWorkspaceBounds "
            "default_planner_request_adapters/FixStartStateBounds "
            "default_planner_request_adapters/FixStartStateCollision "
            "default_planner_request_adapters/FixStartStatePathConstraints",
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl["move_group"].update(load_yaml("franka_moveit_config", "config/ompl_planning.yaml"))

    controllers = {
        # same controller names and action namespaces as the real robot (panda_controllers.yaml)
        "moveit_simple_controller_manager": load_yaml("franka_moveit_config", "config/panda_controllers.yaml"),
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.5,
        "trajectory_execution.allowed_goal_duration_margin": 1.0,
        "trajectory_execution.allowed_start_tolerance": 0.05,
    }
    scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("sim_host", default_value=os.environ.get("AL_BRIDGE_HOST", "127.0.0.1")),
            DeclareLaunchArgument("sim_port", default_value=os.environ.get("AL_BRIDGE_PORT", "45210")),
            Node(
                package="al_ros_bridge",
                executable="isaac_bridge",
                output="screen",
                parameters=[
                    {
                        "sim_host": LaunchConfiguration("sim_host"),
                        "sim_port": ParameterValue(LaunchConfiguration("sim_port"), value_type=int),
                    }
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[robot_description, sim_time],
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    kinematics,
                    ompl,
                    execution,
                    controllers,
                    scene_monitor,
                    sim_time,
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="log",
                arguments=["-d", os.path.join(get_package_share_directory("al_ros_bridge"), "config", "moveit.rviz")],
                parameters=[robot_description, robot_description_semantic, ompl, kinematics, sim_time],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
