"""ROS 2 side of the Isaac Lab bridge: makes the simulated Panda look like the real one to MoveIt.

Connects as a TCP client to ``scripts/teleop_moveit.py`` (see ``panda_al/sim_bridge.py`` for the
protocol) and provides the interfaces of the multipanda_ros2 driver stack:

* publishes ``/joint_states`` and ``/clock`` (simulation time)
* action ``/panda_arm_controller/follow_joint_trajectory`` (control_msgs/FollowJointTrajectory)
* action ``/panda_gripper/gripper_action`` (control_msgs/GripperCommand)

plus helpers for data collection:

* services ``/al/episode/{success,failure,discard}`` and ``/al/gripper/{open,close}`` (std_srvs/Trigger)
* the table as a MoveIt collision object, and the cubes as markers on ``/rviz_visual_tools``
"""

from __future__ import annotations

import itertools
import json
import socket
import threading
import time

import rclpy
from builtin_interfaces.msg import Time
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

CUBE_SIZE = 0.0468
CUBE_COLORS = {"cube_1": (0.1, 0.3, 1.0), "cube_2": (1.0, 0.1, 0.1)}


class SimLink:
    """Thread-safe TCP client with request/response matching by message id."""

    def __init__(self, host: str, port: int, on_state, logger):
        self._addr = (host, port)
        self._on_state = on_state
        self._log = logger
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, list]] = {}
        self._ids = itertools.count(1)
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _run(self):
        while True:
            try:
                sock = socket.create_connection(self._addr, timeout=2.0)
            except OSError:
                time.sleep(1.0)
                continue
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            self._log.info(f"connected to Isaac Lab at {self._addr[0]}:{self._addr[1]}")
            buffer = b""
            try:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line.strip():
                            self._dispatch(json.loads(line))
            except OSError:
                pass
            self._sock = None
            sock.close()
            self._log.warn("lost connection to Isaac Lab, reconnecting ...")
            for event, _ in list(self._pending.values()):
                event.set()  # wake waiters; they see an empty result

    def _dispatch(self, msg: dict):
        if msg.get("type") == "state":
            self._on_state(msg)
        elif msg.get("id") in self._pending:
            event, box = self._pending[msg["id"]]
            box.append(msg)
            event.set()

    def send(self, msg: dict) -> bool:
        sock = self._sock
        if sock is None:
            return False
        with self._send_lock:
            try:
                sock.sendall((json.dumps(msg) + "\n").encode())
                return True
            except OSError:
                return False

    def request(self, msg: dict, keep_waiting=lambda: True, on_wait=None) -> dict | None:
        """Send ``msg`` with a fresh id and block until the matching reply (``None`` if the link drops)."""
        msg["id"] = next(self._ids)
        event, box = threading.Event(), []
        self._pending[msg["id"]] = (event, box)
        try:
            if not self.send(msg):
                return None
            while not event.wait(0.05):
                if on_wait is not None:
                    on_wait(msg["id"])
                if not keep_waiting():
                    return None
            return box[0] if box else None
        finally:
            self._pending.pop(msg["id"], None)


class IsaacBridgeNode(Node):
    def __init__(self):
        super().__init__("al_isaac_bridge", parameter_overrides=[Parameter("use_sim_time", value=False)])
        self.declare_parameter("sim_host", "127.0.0.1")
        self.declare_parameter("sim_port", 45210)
        self.declare_parameter("arm_controller", "panda_arm_controller")
        self.declare_parameter("gripper_controller", "panda_gripper")
        self.declare_parameter("base_frame", "panda_link0")
        # table top is at z = 0 in the robot base frame; keep the robot's own footprint out of the box
        self.declare_parameter("table_center", [0.375, 0.0, -0.026])
        self.declare_parameter("table_size", [0.45, 1.6, 0.05])
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        group = ReentrantCallbackGroup()
        self._base_frame = p("base_frame")
        self._last_objects: dict = {}
        self._episode = -1
        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self._clock_pub = self.create_publisher(Clock, "clock", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "rviz_visual_tools", 10)
        self._scene_pub = self.create_publisher(PlanningScene, "planning_scene", 10)

        ActionServer(
            self,
            FollowJointTrajectory,
            f"{p('arm_controller')}/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=group,
        )
        ActionServer(
            self,
            GripperCommand,
            f"{p('gripper_controller')}/gripper_action",
            execute_callback=self._execute_gripper,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            callback_group=group,
        )
        for outcome in ("success", "failure", "discard"):
            self.create_service(Trigger, f"al/episode/{outcome}", self._episode_cb(outcome), callback_group=group)
        self.create_service(Trigger, "al/gripper/open", self._gripper_cb(0.04), callback_group=group)
        self.create_service(Trigger, "al/gripper/close", self._gripper_cb(0.0), callback_group=group)
        self.create_timer(2.0, self._publish_table, callback_group=group)

        self._link = SimLink(p("sim_host"), p("sim_port"), self._on_state, self.get_logger())
        self.get_logger().info(f"waiting for Isaac Lab at {p('sim_host')}:{p('sim_port')} ...")

    # -- sim -> ROS

    def _on_state(self, msg: dict):
        stamp = Time(sec=int(msg["time"]), nanosec=int((msg["time"] % 1.0) * 1e9))
        self._clock_pub.publish(Clock(clock=stamp))
        js = JointState(name=msg["names"], position=msg["position"], velocity=msg["velocity"])
        js.header.stamp = stamp
        self._joint_pub.publish(js)
        self._last_objects = msg.get("objects", {})
        self._publish_markers(stamp)
        if msg.get("episode") != self._episode:
            self._episode = msg.get("episode")
            self.get_logger().info(f"episode {self._episode} started")

    def _publish_markers(self, stamp: Time):
        markers = MarkerArray()
        for i, (name, pose) in enumerate(sorted(self._last_objects.items())):
            m = Marker(ns="isaac_objects", id=i, type=Marker.CUBE, action=Marker.ADD)
            m.header.frame_id = self._base_frame
            m.header.stamp = stamp
            m.pose = _pose(pose)
            m.scale.x = m.scale.y = m.scale.z = CUBE_SIZE
            m.color.r, m.color.g, m.color.b = CUBE_COLORS.get(name, (0.8, 0.8, 0.8))
            m.color.a = 1.0
            markers.markers.append(m)
        self._marker_pub.publish(markers)

    def _publish_table(self):
        center, size = self.get_parameter("table_center").value, self.get_parameter("table_size").value
        table = CollisionObject(id="table", operation=CollisionObject.ADD)
        table.header.frame_id = self._base_frame
        table.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size))]
        table.primitive_poses = [_pose([*center, 0.0, 0.0, 0.0, 1.0])]
        self._scene_pub.publish(PlanningScene(is_diff=True, world=PlanningSceneWorld(collision_objects=[table])))

    # -- ROS -> sim

    def _execute_trajectory(self, goal_handle):
        traj = goal_handle.request.trajectory
        msg = {
            "type": "trajectory",
            "joint_names": list(traj.joint_names),
            "points": [
                {"t": pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9, "positions": list(pt.positions)}
                for pt in traj.points
            ],
        }
        canceled = []

        def on_wait(goal_id):
            if goal_handle.is_cancel_requested and not canceled:
                canceled.append(True)
                self._link.send({"type": "cancel", "id": goal_id})

        reply = self._link.request(msg, on_wait=on_wait)
        result = FollowJointTrajectory.Result()
        status = reply["status"] if reply else "aborted"
        if status == "succeeded":
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
        elif status == "canceled":
            goal_handle.canceled()
        else:
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = reply.get("error", "") if reply else "no connection to Isaac Lab"
            goal_handle.abort()
        return result

    def _execute_gripper(self, goal_handle):
        cmd = goal_handle.request.command
        reply = self._link.request({"type": "gripper", "position": cmd.position, "max_effort": cmd.max_effort})
        result = GripperCommand.Result()
        if reply is None:
            goal_handle.abort()
            return result
        result.position = reply["position"]
        result.stalled = reply["stalled"]
        result.reached_goal = reply["reached_goal"]
        # holding an object (stalled) is a successful grasp, as with the real franka_gripper
        goal_handle.succeed()
        return result

    def _episode_cb(self, outcome: str):
        def cb(_, response):
            reply = self._link.request({"type": "episode", "cmd": outcome})
            response.success = reply is not None
            response.message = json.dumps(reply) if reply else "no connection to Isaac Lab"
            return response

        return cb

    def _gripper_cb(self, position: float):
        def cb(_, response):
            reply = self._link.request({"type": "gripper", "position": position, "max_effort": 0.0})
            response.success = reply is not None
            response.message = json.dumps(reply) if reply else "no connection to Isaac Lab"
            return response

        return cb


def _pose(p) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = p[0], p[1], p[2]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = p[3], p[4], p[5], p[6]
    return pose


def main():
    rclpy.init()
    node = IsaacBridgeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
