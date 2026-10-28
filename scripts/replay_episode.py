"""Replay actions from the exact initial state of recorded episodes and record the result.

Use cases:

* sanity check: replay an episode's own actions and confirm the outcome is reproduced;
* **validate synthetic episodes**: execute an action sequence proposed by Cosmos (``--actions``,
  a ``(T, 8)`` ``.npy`` in the same format as ``data.npz["action"]``) from the failed episode's
  initial state. The result is recorded as a new episode with ``parent_episode`` set, and the
  simulator decides whether it is really a success.

.. code-block:: bash

    isaaclab -p scripts/replay_episode.py --episodes $AL_DATA_DIR/episodes/scripted/failure/*
    isaaclab -p scripts/replay_episode.py --episodes <failed_episode_dir> --actions cosmos_actions.npy \\
        --source synthetic --run_name cosmos_iter1
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default=None, help="Environment id (default: the one the episodes were recorded in).")
parser.add_argument("--episodes", nargs="+", required=True, help="Episode directories to replay.")
parser.add_argument("--actions", default=None, help="Optional (T, 8) .npy to execute instead of the recorded actions.")
parser.add_argument("--run_name", default="replay", help="Episodes go to $AL_DATA_DIR/episodes/<run_name>.")
parser.add_argument("--source", default="replay", help="Value of meta.json 'source' for the recorded episodes.")
parser.add_argument(
    "--hold_time", type=float, default=2.0, help="Hold the last action this long after the sequence [s]."
)
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

if args_cli.task is None:
    recorded = {json.loads((Path(p) / "meta.json").read_text())["task"] for p in args_cli.episodes}
    if len(recorded) != 1:
        parser.error(f"the episodes were recorded in different environments {sorted(recorded)}: pass --task")
    args_cli.task = recorded.pop()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import panda_al.tasks  # noqa: F401
from panda_al.dashcam import get_dashcam
from panda_al.episode_io import load_episode, unflatten_state


def _to_torch(state, device):
    return {
        k: _to_torch(v, device) if isinstance(v, dict) else torch.as_tensor(v, device=device) for k, v in state.items()
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, overrides=hydra_overrides)
    env_cfg.recorders.dashcam.output_dir = os.path.join(
        os.environ.get("AL_DATA_DIR", "data"), "episodes", args_cli.run_name
    )
    env_cfg.recorders.dashcam.source = args_cli.source

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        dashcam = get_dashcam(env)
        env_ids = torch.tensor([0], device=env.device)
        results = []

        for path in args_cli.episodes:
            episode = load_episode(path)
            if not episode.initial_state:
                print(f"[replay] skipping {path}: no initial_state.npz")
                continue
            if args_cli.actions is not None:
                actions = np.load(args_cli.actions).astype(np.float32)
            else:
                actions = episode.data["action"][~episode.data["is_last"]]
            hold_steps = int(args_cli.hold_time / env.step_dt)

            env.reset_to(_to_torch(unflatten_state(episode.initial_state), env.device), env_ids, is_relative=True)
            task_info = episode.meta.extra.get("task")
            if task_info is not None and hasattr(env, "restore_episode_info"):
                env.restore_episode_info([0], [task_info])  # multi-task env: task, prompt and goal of the episode
            dashcam.parent_episode[0] = episode.meta.episode_id
            dashcam.episode_extra[0] = {"replayed_from": str(path), "actions_file": args_cli.actions}
            done = False
            for t in range(len(actions) + hold_steps):
                action = torch.as_tensor(actions[min(t, len(actions) - 1)], device=env.device).unsqueeze(0)
                _, _, terminated, truncated, _ = env.step(action)
                if bool(terminated[0] | truncated[0]):
                    done = True
                    break
            if not done:
                # sequence exhausted without success: end the episode explicitly
                dashcam.outcome_override[0] = "failure"
                env.reset()
            outcome, out_path = dashcam.last_finished[0]
            results.append(
                {
                    "source_episode": str(path),
                    "original_outcome": episode.meta.outcome,
                    "replay_outcome": outcome,
                    "replay_path": out_path,
                }
            )
            print(f"[replay] {episode.meta.episode_id}: original={episode.meta.outcome} replay={outcome}")

        print(json.dumps(results, indent=2))
        env.close()


if __name__ == "__main__":
    main()
