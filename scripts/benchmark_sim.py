"""Simulator throughput for a given number of environments (milestone M0 of docs/automated_collection.md).

One process measures one env count (Isaac Lab does not like several env instances per process); loop in the shell:

.. code-block:: bash

    for n in 1 4 8 16 32; do ./al.sh run isaaclab -p scripts/benchmark_sim.py --num_envs $n; done
    ./al.sh run isaaclab -p scripts/benchmark_sim.py --num_envs 16 --record     # with the dashcam writing videos

The scripted expert drives the robots (a realistic load: IK, resets, contacts). Results are printed and appended as
JSON lines to ``$AL_DATA_DIR/benchmarks/sim.jsonl``. ``--record`` writes episodes to a throw-away directory.
"""

import argparse
import json
import os
import tempfile
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="PandaAL-MultiTask-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300, help="Measured control steps (after the warm-up).")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--record", action="store_true", help="Keep the dashcam on (video encoding, disk writes).")
parser.add_argument("--profile", action="store_true", help="With --record: time the dashcam's parts per control step.")
parser.add_argument(
    "--defer", action="store_true", help="With --record: collection mode (buffer frames, encode only failures)."
)
parser.add_argument("--cameras", nargs="+", default=None, help="Keep only these cameras in the scene (default: all four).")
parser.add_argument("--depth", action="store_true", help="Record the wrist depth stream too (default: off).")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import panda_al.tasks  # noqa: F401
from panda_al.dashcam import get_dashcam
from panda_al.scripted import Corruption, ScriptedPickPlace
from panda_al.tasks.pick_place_env_cfg import PICK_OBJECT, PLACE_OBJECT, restrict_cameras


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    scratch = tempfile.mkdtemp(prefix="al_bench_")
    env_cfg.recorders.dashcam.output_dir = scratch
    env_cfg.recorders.dashcam.enabled = args_cli.record
    if args_cli.cameras is not None:
        restrict_cameras(env_cfg, args_cli.cameras)  # removes the other cameras from the scene altogether
    if not args_cli.depth:
        env_cfg.recorders.dashcam.depth_camera_names = []
    if args_cli.defer:
        env_cfg.recorders.dashcam.defer_encoding = True
        env_cfg.recorders.dashcam.save_outcomes = ["failure"]

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        dashcam = get_dashcam(env)
        expert = ScriptedPickPlace(env, PICK_OBJECT, PLACE_OBJECT)
        all_ids = list(range(env.num_envs))
        expert.reset(all_ids, [Corruption() for _ in all_ids])

        def step() -> int:
            _, _, terminated, truncated, _ = env.step(expert.compute())
            done = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).tolist()
            if done:
                expert.reset(done, [Corruption() for _ in done])
                dashcam.snapshot_initial_state(done)
            return len(done)

        timers = {"dashcam_append_rows": 0.0, "writer_append(queue put)": 0.0}
        if args_cli.profile:
            from panda_al import dashcam as dashcam_module
            from panda_al.episode_io import EpisodeWriter

            def timed(obj, name, key):
                original = getattr(obj, name)

                def wrapper(*a, **k):
                    t0 = time.perf_counter()
                    out = original(*a, **k)
                    timers[key] += time.perf_counter() - t0
                    return out

                setattr(obj, name, wrapper)

            timed(dashcam_module.DashcamRecorder, "_append_rows", "dashcam_append_rows")
            timed(EpisodeWriter, "append", "writer_append(queue put)")

        for _ in range(args_cli.warmup):
            step()
        for key in timers:
            timers[key] = 0.0
        torch.cuda.synchronize()
        episodes = 0
        start = time.perf_counter()
        for _ in range(args_cli.steps):
            episodes += step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        free, total = torch.cuda.mem_get_info()
        result = {
            "task": args_cli.task,
            "num_envs": env.num_envs,
            "record": args_cli.record,
            "defer": args_cli.defer,
            "cameras": args_cli.cameras or "all",
            "depth": args_cli.depth,
            "control_steps": args_cli.steps,
            "ms_per_control_step": round(1000 * elapsed / args_cli.steps, 1),
            "env_steps_per_s": round(env.num_envs * args_cli.steps / elapsed, 1),
            "episodes_per_min": round(60 * episodes / elapsed, 2),
            "gpu_mem_used_gb": round((total - free) / 2**30, 2),
            "device": torch.cuda.get_device_name(),
        }
        if args_cli.profile:
            result["ms_per_control_step_in"] = {k: round(1000 * v / args_cli.steps, 1) for k, v in timers.items()}
        print("[benchmark] " + json.dumps(result))
        out = os.path.join(os.environ.get("AL_DATA_DIR", "data"), "benchmarks")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "sim.jsonl"), "a") as f:
            f.write(json.dumps({**result, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        env.close()


if __name__ == "__main__":
    main()
