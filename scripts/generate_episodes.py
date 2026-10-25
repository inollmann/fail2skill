"""Generate episodes automatically with the scripted expert, optionally corrupted so they fail.

Every episode is recorded by the dashcam and filed under success/ or failure/ according to the
simulator's success check. The corruption that was injected (if any) is stored in meta.json under
``extra.corruption``.

.. code-block:: bash

    # 200 episodes, 80 % of them corrupted with a random failure mode, 8 envs in parallel
    isaaclab -p scripts/generate_episodes.py --num_envs 8 --num_episodes 200 --corrupt_ratio 0.8

    # clean demonstrations only
    isaaclab -p scripts/generate_episodes.py --num_episodes 50 --corrupt_ratio 0

    # only some failure modes
    isaaclab -p scripts/generate_episodes.py --num_episodes 50 --modes miss_grasp drop_in_transit
"""

import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

from panda_al.scripted import CORRUPTION_MODES

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", default="PandaAL-PickPlace-v0")
parser.add_argument("--run_name", default="scripted", help="Episodes go to $AL_DATA_DIR/episodes/<run_name>.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20, help="Stop after this many finished episodes.")
parser.add_argument("--corrupt_ratio", type=float, default=0.8, help="Fraction of episodes with an injected failure.")
parser.add_argument("--modes", nargs="+", default=sorted(CORRUPTION_MODES), choices=sorted(CORRUPTION_MODES))
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import panda_al.tasks  # noqa: F401
from panda_al.dashcam import get_dashcam
from panda_al.scripted import Corruption, ScriptedPickPlace, sample_corruption
from panda_al.tasks.pick_place_env_cfg import PICK_OBJECT, PLACE_OBJECT


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides
    )
    env_cfg.seed = args_cli.seed
    env_cfg.recorders.dashcam.output_dir = os.path.join(
        os.environ.get("AL_DATA_DIR", "data"), "episodes", args_cli.run_name
    )
    env_cfg.recorders.dashcam.source = "scripted"

    gen = torch.Generator().manual_seed(args_cli.seed)

    def new_corruption() -> Corruption:
        if torch.rand(1, generator=gen).item() >= args_cli.corrupt_ratio:
            return Corruption()
        return sample_corruption(args_cli.modes, gen)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        dashcam = get_dashcam(env)
        expert = ScriptedPickPlace(env, PICK_OBJECT, PLACE_OBJECT)

        def start(env_ids: list[int]):
            corruptions = [new_corruption() for _ in env_ids]
            expert.reset(env_ids, corruptions)
            for env_id, corruption in zip(env_ids, corruptions, strict=True):
                dashcam.episode_extra[env_id] = {
                    "generator": "scripted",
                    "seed": args_cli.seed,
                    "corruption": corruption.to_dict(),
                }

        start(list(range(env.num_envs)))
        finished = {"success": 0, "failure": 0}
        while env.sim.is_running() and sum(finished.values()) < args_cli.num_episodes:
            _, _, terminated, truncated, _ = env.step(expert.compute())
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).tolist()
            if done_ids:
                for env_id in done_ids:
                    outcome = dashcam.last_finished[env_id][0]
                    finished[outcome] = finished.get(outcome, 0) + 1
                print(f"[generate] {sum(finished.values())}/{args_cli.num_episodes} finished: {finished}")
                start(done_ids)
        env.close()


if __name__ == "__main__":
    main()
