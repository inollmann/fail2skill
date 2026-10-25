"""Single-key operator console for teleop data collection (run with ``ros2 run al_ros_bridge al_episode``)."""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

KEYS = {
    "o": ("al/gripper/open", "open gripper"),
    "c": ("al/gripper/close", "close gripper"),
    "s": ("al/episode/success", "end episode, label SUCCESS"),
    "f": ("al/episode/failure", "end episode, label FAILURE"),
    "d": ("al/episode/discard", "end episode, DISCARD it"),
}

HELP = "\n".join(f"  {key}  {desc}" for key, (_, desc) in KEYS.items()) + "\n  q  quit"


def main():
    rclpy.init()
    node = Node("al_episode_cli")
    clients = {key: node.create_client(Trigger, srv) for key, (srv, _) in KEYS.items()}
    print("Episode console. Keys:\n" + HELP, flush=True)
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                break
            if key not in clients:
                print(HELP, flush=True)
                continue
            client = clients[key]
            if not client.wait_for_service(timeout_sec=1.0):
                print("bridge not available (is al_isaac_bridge running?)", flush=True)
                continue
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
            reply = future.result()
            message = reply.message if reply else "timeout"
            print(f"[{KEYS[key][1]}] {message}", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
