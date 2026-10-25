"""Simulator side of the ROS 2 / MoveIt bridge.

The Isaac Lab process does not import ROS. It runs a tiny TCP server speaking line-delimited JSON;
the ``al_ros_bridge`` node in the ROS 2 Humble container connects to it and exposes the same
interfaces as the real Panda driver stack (multipanda_ros2):

* ``/joint_states`` and ``/clock`` (sim time)
* ``/panda_arm_controller/follow_joint_trajectory`` (control_msgs/FollowJointTrajectory)
* ``/panda_gripper/gripper_action`` (control_msgs/GripperCommand)

so ``move_group`` with ``franka_moveit_config`` cannot tell the simulator from the robot.
Keeping ROS out of the Kit process avoids the Python 3.12 (Isaac Sim) vs 3.10 (Humble) mismatch.

Messages (one JSON object per line)::

    sim -> ros: {"type": "state", "time": t, "names": [...], "position": [...], "velocity": [...],
                 "objects": {"cube_1": [x, y, z, qx, qy, qz, qw], ...}, "episode": n}
                {"type": "trajectory_result", "id": i, "status": "succeeded" | "aborted" | "canceled", "error": str}
                {"type": "gripper_result", "id": i, "position": p, "reached_goal": bool, "stalled": bool}
                {"type": "episode_result", "id": i, "outcome": str, "path": str | null}
    ros -> sim: {"type": "trajectory", "id": i, "joint_names": [...], "points": [{"t": s, "positions": [...]}, ...]}
                {"type": "cancel", "id": i}
                {"type": "gripper", "id": i, "position": p, "max_effort": f}
                {"type": "episode", "id": i, "cmd": "success" | "failure" | "discard"}
"""

from __future__ import annotations

import json
import select
import socket

import numpy as np


class BridgeServer:
    """Single-client, non-blocking line-delimited JSON server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 45210):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(1)
        self._listener.setblocking(False)
        self._conn: socket.socket | None = None
        self._buffer = b""
        print(f"[bridge] listening on {host}:{port}")

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def poll(self) -> list[dict]:
        """Accept a pending client and return all complete messages received since the last call."""
        if self._conn is None:
            try:
                self._conn, addr = self._listener.accept()
                self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._buffer = b""
                print(f"[bridge] client connected from {addr}")
            except BlockingIOError:
                return []
        messages = []
        while self._conn is not None and select.select([self._conn], [], [], 0)[0]:
            try:
                chunk = self._conn.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                self._drop()
                break
            self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if line.strip():
                messages.append(json.loads(line))
        return messages

    def send(self, message: dict):
        if self._conn is None:
            return
        try:
            self._conn.sendall((json.dumps(message) + "\n").encode())
        except OSError:
            self._drop()

    def _drop(self):
        print("[bridge] client disconnected")
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    def close(self):
        if self._conn is not None:
            self._conn.close()
        self._listener.close()


class TrajectoryExecutor:
    """Plays back a joint trajectory against simulation time by linear interpolation."""

    def __init__(self, joint_names: list[str], settle_time: float = 0.25):
        self.joint_names = list(joint_names)
        self.settle_time = settle_time
        self.goal_id: int | None = None
        self._times = np.zeros(0)
        self._positions = np.zeros((0, len(joint_names)))
        self._start = 0.0

    @property
    def active(self) -> bool:
        return self.goal_id is not None

    def start(self, msg: dict, now: float, hold: np.ndarray):
        """Start ``msg`` (a ``trajectory`` message); joints missing from the message keep ``hold``."""
        index = {name: i for i, name in enumerate(msg["joint_names"])}
        times, positions = [], []
        for point in msg["points"]:
            q = hold.copy()
            for j, name in enumerate(self.joint_names):
                if name in index:
                    q[j] = point["positions"][index[name]]
            times.append(float(point["t"]))
            positions.append(q)
        if not times or times[0] > 0.0:
            times.insert(0, 0.0)
            positions.insert(0, hold.copy())
        self._times = np.asarray(times)
        self._positions = np.asarray(positions)
        self._start = now
        self.goal_id = msg["id"]

    def target(self, now: float) -> tuple[np.ndarray, bool]:
        """Joint target at ``now`` and whether the trajectory (plus settle time) has finished."""
        t = now - self._start
        q = np.array([np.interp(t, self._times, self._positions[:, j]) for j in range(len(self.joint_names))])
        return q, t >= self._times[-1] + self.settle_time

    def stop(self) -> int | None:
        goal_id, self.goal_id = self.goal_id, None
        return goal_id
