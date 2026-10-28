"""Unattended collection of failure episodes with a served policy (see ``docs/automated_collection.md``).

.. code-block:: bash

    ./al.sh collect configs/campaigns/example.yaml        # restarts the simulator after crashes, resumes from disk

The robots of a multi-task env (``PandaAL-MultiTask-v0``) work through the campaign's tasks, controlled by a policy
server (or, for dry runs and baselines, the scripted expert). Every finished episode is judged by the simulator.
Failures are tagged (stalled_at_start, no_grasp, dropped, misplaced, collision, knocked_off, timeout) and saved subject
to the campaign's quotas; successes are dropped without ever being encoded. Two files in the run directory
``$AL_DATA_DIR/episodes/<name>/`` record the campaign: ``stats.jsonl`` (every episode, for success rates) and
``manifest.jsonl`` (the saved ones). Restarting with the same campaign file continues where it stopped.

Exit status (the ``isaaclab`` launcher turns every exit code into 1, so the status is written to the file named by
``$AL_COLLECT_STATUS_FILE``, which the ``./al.sh collect`` supervisor reads): 0 finished (quotas or limits reached),
2 invalid campaign, 75 policy server lost (restart later). No status file: crash, the supervisor restarts.
"""

import argparse
import os
import shutil
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

from panda_al.collect.campaign import load_campaign


def finish(code: int):
    """Write the exit status for the supervisor and end the process (closing the Kit app would exit with 0)."""
    status_file = os.environ.get("AL_COLLECT_STATUS_FILE")
    if status_file:
        with open(status_file, "w") as f:
            f.write(str(code))
    sys.stdout.flush()
    os._exit(code)


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("campaign", help="Campaign YAML (see configs/campaigns/).")
add_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

try:
    campaign = load_campaign(args_cli.campaign)
except (ValueError, OSError) as err:
    print(f"[collect] invalid campaign: {err}")
    finish(2)

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import panda_al.tasks  # noqa: F401
from panda_al.tasks.pick_place_env_cfg import restrict_cameras
from panda_al.collect.drivers import ExpertDriver, PolicyDriver
from panda_al.collect.ledger import Ledger
from panda_al.collect.tags import failure_tags
from panda_al.dashcam import EpisodeSummary, get_dashcam
from panda_al.obstacle import place_obstacles, task_state_role_pose
from panda_al.policy import ADAPTERS, PolicyUnavailable
from panda_al.scripted import CORRUPTION_MODES, OBSTACLE_MODE, Corruption

EXIT_POLICY_UNAVAILABLE = 75
REPORT_EVERY_S = 30.0


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.exists(os.path.join(root, f)))
    return total / 2**30


def main():
    env_cfg = parse_env_cfg(campaign.env, device=args_cli.device, num_envs=campaign.num_envs, overrides=hydra_overrides)
    env_cfg.seed = campaign.seed
    dashcam_cfg = env_cfg.recorders.dashcam
    dashcam_cfg.output_dir = campaign.output_dir
    dashcam_cfg.source = "policy" if campaign.driver == "policy" else "scripted"
    # cameras outside the campaign (and the policy adapter's) are removed from the scene: every camera in it is rendered
    needed = set(campaign.cameras) | (set(ADAPTERS[campaign.policy.adapter]().camera_names()) if campaign.driver == "policy" else set())
    restrict_cameras(env_cfg, [c for c in env_cfg.camera_names if c in needed])
    dashcam_cfg.camera_names = [c for c in dashcam_cfg.camera_names if c in campaign.cameras]
    dashcam_cfg.depth_camera_names = ["wrist"] if campaign.depth and "wrist" in env_cfg.camera_names else []
    dashcam_cfg.defer_encoding = True
    dashcam_cfg.save_outcomes = list(campaign.save.outcomes)
    if campaign.tasks is not None:
        params = env_cfg.events.reset_task_scene.params
        params["tasks"], params["task_weights"] = list(campaign.tasks), dict(campaign.tasks)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(campaign.env, cfg=env_cfg).unwrapped
        if not hasattr(env, "task_state"):
            print(f"[collect] {campaign.env} has no per-env tasks: use a multi-task env such as PandaAL-MultiTask-v0")
            env.close()
            finish(2)
        shutil.rmtree(os.path.join(campaign.output_dir, "in_progress"), ignore_errors=True)  # left by a crashed run
        env.reset()
        dashcam = get_dashcam(env)
        task_names = [t.name for t in env.task_state.tasks]
        ledger = Ledger(
            campaign.output_dir,
            task_names,
            campaign.save.quota_per_task,
            campaign.save.quota_per_tag,
            campaign.save.drop_tags,
        )
        object_names = list(dashcam_cfg.object_names)
        rng = torch.Generator().manual_seed(campaign.seed + ledger.total_finished)  # a resumed run draws new episodes
        label = campaign.policy.label if campaign.driver == "policy" else "scripted"
        print(f"[collect] {campaign.name}: {campaign.driver} driver, {campaign.num_envs} envs, tasks {task_names}")
        print(f"[collect] resuming with: {ledger.summary()}")

        def episode_hook(summary: EpisodeSummary) -> bool:
            info = summary.meta.extra.get("task", {})
            task = info.get("task", "unknown")
            tags: list[str] = []
            if summary.outcome == "failure" and "failure" not in campaign.save.outcomes:
                kept, reason = False, "failure_not_saved"
            elif summary.outcome == "failure":
                data = summary.data
                if not all(np.isfinite(data[k]).all() for k in ("joint_position", "object_pose")):
                    kept, reason = False, "invalid_numerics"
                else:
                    tags = failure_tags(data, object_names, info, summary.termination_terms)
                    kept, reason = ledger.decide(task, tags)
            elif "success" in campaign.save.outcomes:  # demonstration collection: successes are what is kept
                kept, reason = ledger.decide(task, [])
                reason = "success_kept" if kept else f"success_{reason}"
            else:
                kept = torch.rand(1, generator=rng).item() < campaign.save.keep_success_fraction
                reason = "success_kept" if kept else "success_dropped"
            summary.extra.update({"failure_tags": tags, "campaign": campaign.name, "policy_label": label})
            persist = ledger.record(
                {
                    "episode_id": summary.meta.episode_id,
                    "task": task,
                    "prompt": info.get("prompt"),
                    "pick": info.get("pick"),
                    "place": info.get("place"),
                    "objects": info.get("objects"),
                    "outcome": summary.outcome,
                    "steps": summary.writer.num_rows,
                    "termination_terms": summary.termination_terms,
                    "tags": tags,
                    "policy_label": label,
                    "perturbation": summary.meta.extra.get("perturbation", {}).get("mode"),
                    "path": str(summary.writer.target_path(summary.outcome).relative_to(summary.writer.run_dir)) if kept else None,
                },
                kept,
                reason,
            )
            # the stats / manifest lines of a kept episode are written once it is complete on disk
            summary.writer.on_saved = lambda _path: persist()
            return kept

        dashcam.episode_hook = episode_hook
        try:
            driver = (
                PolicyDriver(env, campaign, dashcam_cfg.instruction)
                if campaign.driver == "policy"
                else ExpertDriver(env, campaign, rng)
            )
        except PolicyUnavailable as err:
            print(f"[collect] {err}; stopping, restart later to continue")
            env.close()
            finish(EXIT_POLICY_UNAVAILABLE)

        if campaign.driver == "policy":
            print(f"[collect] policy server {campaign.policy.host}:{campaign.policy.port}, {driver.mode}")

        def start(env_ids: list[int]):
            extras = driver.start(env_ids)
            corruptions = []
            for env_id in env_ids:
                use_obstacle = torch.rand(1, generator=rng).item() < campaign.obstacle_probability
                corruptions.append(CORRUPTION_MODES[OBSTACLE_MODE](rng) if use_obstacle else Corruption())
            if any(c.mode == OBSTACLE_MODE for c in corruptions):
                place_obstacles(env, env_ids, corruptions, lambda role: task_state_role_pose(env, role))
            for env_id, corruption in zip(env_ids, corruptions, strict=True):
                if corruption.mode == OBSTACLE_MODE:
                    extras[env_id]["perturbation"] = corruption.to_dict()  # includes the obstacle's position
                dashcam.episode_extra[env_id] = extras[env_id]
            dashcam.snapshot_initial_state(env_ids)  # obstacle / expert plan may have changed the scene

        started = time.monotonic()
        last_report = started
        last_disk_check, disk_gb = started, 0.0
        exit_code = 0
        try:
            start(list(range(env.num_envs)))
            while env.sim.is_running():
                now = time.monotonic()
                limits = campaign.limits
                if ledger.quotas_met():
                    print("[collect] all task quotas reached")
                    break
                if limits.max_episodes is not None and ledger.total_finished >= limits.max_episodes:
                    print("[collect] episode limit reached")
                    break
                if limits.max_hours is not None and now - started > 3600 * limits.max_hours:
                    print("[collect] time limit reached")
                    break
                if limits.max_disk_gb is not None and now - last_disk_check > 60:
                    last_disk_check, disk_gb = now, dir_size_gb(campaign.output_dir)
                    if disk_gb > limits.max_disk_gb:
                        print(f"[collect] disk limit reached ({disk_gb:.1f} GB)")
                        break
                actions = torch.as_tensor(driver.actions(), device=env.device)
                _, _, terminated, truncated, _ = env.step(actions)
                done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).tolist()
                if done_ids:
                    start(done_ids)
                if now - last_report > REPORT_EVERY_S:
                    last_report = now
                    print(f"[collect] {(now - started) / 60:.0f} min: {ledger.summary()}")
                    if driver.stats():
                        print(f"[collect] {driver.stats()}")
        except PolicyUnavailable as err:
            print(f"[collect] {err}; stopping, restart later to continue")
            exit_code = EXIT_POLICY_UNAVAILABLE
        finally:
            driver.close()
            env.close()  # discards running episodes and waits for the encoder pool
        print(f"[collect] done: {ledger.summary()}")
        finish(exit_code)  # everything is on disk: env.close() waited for the encoders


if __name__ == "__main__":
    main()
