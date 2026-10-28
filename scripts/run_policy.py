"""Roll out a policy served over websocket (see :mod:`panda_al.policy`) and record every episode.

Start the policy server first (any machine that can reach this one), then:

.. code-block:: bash

    # pi0.5 DROID served by openpi (scripts/serve_policy.py ... pi05_droid) on port 8000
    isaaclab -p scripts/run_policy.py --adapter droid --num_episodes 20 --run_name pi05_droid

    # a LeRobot policy (e.g. SmolVLA) served by `python -m panda_al.policy.serve lerobot ...`
    isaaclab -p scripts/run_policy.py --adapter native --host <server> --run_name smolvla_iter0

Episodes are recorded by the dashcam with ``source = "policy"``; the server metadata, adapter settings and
policy name are stored in ``meta.json`` under ``extra.policy``. Each episode opens a new connection, which
resets stateful policies on our server.
"""

import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

from panda_al.policy.adapters import ADAPTERS

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="PandaAL-PickPlace-v0")
parser.add_argument("--host", default=os.environ.get("AL_POLICY_HOST", "127.0.0.1"))
parser.add_argument("--port", type=int, default=int(os.environ.get("AL_POLICY_PORT", "8000")))
parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="native")
parser.add_argument("--open_loop_horizon", type=int, default=None, help="Actions executed per query (default: adapter's).")
parser.add_argument("--instruction", default=None, help="Override the task's language instruction.")
parser.add_argument("--run_name", default="policy", help="Episodes go to $AL_DATA_DIR/episodes/<run_name>.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import panda_al.tasks  # noqa: F401
from panda_al.collect.drivers import read_observations
from panda_al.dashcam import get_dashcam
from panda_al.policy import PolicyRunner


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders.dashcam.output_dir = os.path.join(
        os.environ.get("AL_DATA_DIR", "data"), "episodes", args_cli.run_name
    )
    env_cfg.recorders.dashcam.source = "policy"
    if args_cli.instruction:
        env_cfg.recorders.dashcam.instruction = args_cli.instruction
    instruction = env_cfg.recorders.dashcam.instruction

    adapter = ADAPTERS[args_cli.adapter]()
    if args_cli.open_loop_horizon is not None:
        adapter.open_loop_horizon = args_cli.open_loop_horizon

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        dashcam = get_dashcam(env)
        robot = env.scene["robot"]
        arm_ids = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        finger_id = robot.find_joints("panda_finger_joint1")[0][0]
        runners = [PolicyRunner(adapter, args_cli.host, args_cli.port) for _ in range(env.num_envs)]

        def start(env_ids: list[int]):
            for env_id in env_ids:
                metadata = runners[env_id].start_episode()
                dashcam.episode_extra[env_id] = {
                    "policy": {"server": runners[env_id].uri, "metadata": metadata, **adapter.describe()}
                }

        start(list(range(env.num_envs)))
        print(f"[policy] connected to {runners[0].uri}: {runners[0].client.metadata}")
        finished = {"success": 0, "failure": 0}
        while env.sim.is_running() and sum(finished.values()) < args_cli.num_episodes:
            observations = read_observations(env, adapter.camera_names(), arm_ids, finger_id, instruction)
            actions = np.stack([runner.act(obs) for runner, obs in zip(runners, observations, strict=True)])
            _, _, terminated, truncated, _ = env.step(torch.as_tensor(actions, device=env.device))
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).tolist()
            if done_ids:
                for env_id in done_ids:
                    outcome = dashcam.last_finished[env_id][0]
                    finished[outcome] = finished.get(outcome, 0) + 1
                print(f"[policy] {sum(finished.values())}/{args_cli.num_episodes} finished: {finished}")
                start(done_ids)
        for runner in runners:
            runner.close()
        env.close()


if __name__ == "__main__":
    main()
