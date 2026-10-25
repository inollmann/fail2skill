"""Record episodes by driving the simulated Panda with MoveIt 2 (through the al_ros_bridge node).

Run this in the ``sim`` container, then start the ``ros`` container (MoveIt + RViz + bridge).
Plan and execute in RViz; every episode is recorded by the dashcam. Episodes end when the task
succeeds, when they time out, or when the operator ends them (``al_episode`` CLI in the ros
container: s = success, f = failure, d = discard).

.. code-block:: bash

    isaaclab -p scripts/teleop_moveit.py --viz kit    # with the Isaac Sim window
    isaaclab -p scripts/teleop_moveit.py              # headless (RViz only)
"""

import argparse
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="PandaAL-PickPlace-v0")
parser.add_argument("--run_name", default="teleop", help="Episodes go to $AL_DATA_DIR/episodes/<run_name>.")
parser.add_argument("--host", default="0.0.0.0", help="Bridge bind address.")
parser.add_argument("--port", type=int, default=int(os.environ.get("AL_BRIDGE_PORT", 45210)))
parser.add_argument("--episode_length", type=float, default=120.0, help="Episode time limit [s].")
parser.add_argument("--no_realtime", action="store_true", help="Run as fast as possible instead of 1x real time.")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import panda_al.tasks  # noqa: F401
from panda_al.dashcam import get_dashcam
from panda_al.sim_bridge import BridgeServer, TrajectoryExecutor
from panda_al.tasks.pick_place_env_cfg import PICK_OBJECT, PLACE_OBJECT

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_SETTLE = 0.6  # [s] sim time to wait before reporting a gripper command as done


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, overrides=hydra_overrides)
    env_cfg.episode_length_s = args_cli.episode_length
    env_cfg.recorders.dashcam.output_dir = os.path.join(
        os.environ.get("AL_DATA_DIR", "data"), "episodes", args_cli.run_name
    )
    env_cfg.recorders.dashcam.source = "teleop"

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        dashcam = get_dashcam(env)
        robot = env.scene["robot"]
        arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
        finger_id = robot.find_joints("panda_finger_joint1")[0][0]
        objects = {name: env.scene[name] for name in (PICK_OBJECT, PLACE_OBJECT)}

        server = BridgeServer(args_cli.host, args_cli.port)
        traj = TrajectoryExecutor(ARM_JOINTS)
        dt = env.step_dt
        sim_time = 0.0  # monotonic across episodes (published as /clock)
        episode = 0
        gripper_cmd = 0.0
        gripper_goal: tuple[int, float] | None = None  # (goal id, sim time when done)

        def reset_hold():
            nonlocal gripper_cmd
            hold = robot.data.joint_pos.torch[0, arm_ids].cpu().numpy().astype(np.float64)
            gripper_cmd = 0.0
            return hold

        def abort_goals(reason: str):
            nonlocal gripper_goal
            goal_id = traj.stop()
            if goal_id is not None:
                server.send({"type": "trajectory_result", "id": goal_id, "status": "aborted", "error": reason})
            if gripper_goal is not None:
                server.send(
                    {
                        "type": "gripper_result",
                        "id": gripper_goal[0],
                        "position": 0.0,
                        "reached_goal": False,
                        "stalled": False,
                    }
                )
                gripper_goal = None

        hold = reset_hold()
        print("[teleop] ready - start the ros container to connect MoveIt")
        while env.sim.is_running():
            wall_start = time.monotonic()

            # handle bridge requests
            for msg in server.poll():
                kind = msg.get("type")
                if kind == "trajectory":
                    abort_goals("preempted by a new trajectory")
                    traj.start(msg, sim_time, hold)
                elif kind == "cancel" and traj.goal_id == msg.get("id"):
                    traj.stop()
                    server.send({"type": "trajectory_result", "id": msg["id"], "status": "canceled", "error": ""})
                elif kind == "gripper":
                    gripper_cmd = 1.0 if float(msg["position"]) < 0.02 else 0.0
                    gripper_goal = (msg["id"], sim_time + GRIPPER_SETTLE)
                elif kind == "episode":
                    cmd = msg.get("cmd", "failure")
                    dashcam.outcome_override[0] = cmd
                    abort_goals(f"episode ended by operator ({cmd})")
                    env.reset()
                    dashcam.outcome_override.pop(0, None)  # unused if the episode had no steps yet
                    outcome, path = dashcam.last_finished.get(0, (cmd, None))
                    server.send({"type": "episode_result", "id": msg.get("id"), "outcome": outcome, "path": path})
                    episode += 1
                    hold = reset_hold()

            # compute the action
            if traj.active:
                hold, finished = traj.target(sim_time)
                if finished:
                    server.send({"type": "trajectory_result", "id": traj.stop(), "status": "succeeded", "error": ""})
            action = torch.tensor([[*hold, gripper_cmd]], dtype=torch.float32, device=env.device)
            _, _, terminated, truncated, _ = env.step(action)
            sim_time += dt

            if bool(terminated[0] | truncated[0]):
                outcome, _ = dashcam.last_finished.get(0, ("?", None))
                abort_goals(f"episode ended ({outcome})")
                episode += 1
                hold = reset_hold()

            if gripper_goal is not None and sim_time >= gripper_goal[1]:
                finger = float(robot.data.joint_pos.torch[0, finger_id])
                stalled = gripper_cmd > 0.5 and finger > 0.002  # closed on an object
                server.send(
                    {
                        "type": "gripper_result",
                        "id": gripper_goal[0],
                        "position": finger,
                        "reached_goal": not stalled,
                        "stalled": stalled,
                    }
                )
                gripper_goal = None

            # publish state
            q = robot.data.joint_pos.torch[0].cpu().numpy()
            qd = robot.data.joint_vel.torch[0].cpu().numpy()
            finger = float(q[finger_id])
            origin = env.scene.env_origins[0]
            server.send(
                {
                    "type": "state",
                    "time": sim_time,
                    "names": ARM_JOINTS + FINGER_JOINTS,
                    "position": [float(v) for v in q[arm_ids]] + [finger, finger],
                    "velocity": [float(v) for v in qd[arm_ids]] + [0.0, 0.0],
                    "objects": {
                        name: (
                            torch.cat([obj.data.root_pos_w.torch[0] - origin, obj.data.root_quat_w.torch[0]])
                        ).tolist()
                        for name, obj in objects.items()
                    },
                    "episode": episode,
                }
            )

            if not args_cli.no_realtime:
                time.sleep(max(0.0, dt - (time.monotonic() - wall_start)))

        server.close()
        env.close()


if __name__ == "__main__":
    main()
