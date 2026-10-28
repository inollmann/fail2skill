"""Episodes -> LeRobotDataset (v3), for fine-tuning SmolVLA / ACT / pi0 with LeRobot.

.. code-block:: bash

    ./al.sh export-lerobot --source data/episodes/demos --out data/lerobot/demos_v0
    # several sources, the second one counted twice (emphasis; M5 uses this for the weak tasks)
    ./al.sh export-lerobot --source data/episodes/demos --source data/episodes/cosmos_iter1:2 --out data/lerobot/mix_v1

Needs ``lerobot`` with the dataset extra (the policy image has it); not Isaac Lab.

What a frame contains (the DROID convention of the episode format: row ``t`` pairs the observation before
``action[t]`` with that action; the terminal row has no valid action and is dropped):

``observation.images.<camera>``  video, the recorded camera frames
``observation.state``            (8,) 7 arm joint positions [rad] + gripper closure (0 open .. 1 closed)
``action``                       (8,) 7 absolute joint position targets [rad] + gripper command (0 open, 1 closed)
``task``                         the episode's language prompt

Each source is a run directory (``success/`` and ``failure/`` below it); ``--outcomes`` picks which are used (default:
success only). ``export_info.json`` in the output lists every exported episode and its repeat count.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..episode_io import Episode, iter_episodes, load_episode

STATE_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["gripper_closure"]
ACTION_NAMES = [f"panda_joint{i}_target" for i in range(1, 8)] + ["gripper_command"]


def parse_source(text: str) -> tuple[Path, int]:
    """``path`` or ``path:repeat``."""
    path, _, repeat = text.rpartition(":")
    if path and repeat.isdigit() and not Path(text).exists():
        return Path(path), max(1, int(repeat))
    return Path(text), 1


def episode_arrays(episode: Episode, cameras: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Per-frame images (T-1, H, W, 3) per camera, state (T-1, 8) and action (T-1, 8) of an episode."""
    data = episode.data
    keep = ~data["is_last"].astype(bool)  # the terminal observation has no action
    state = np.concatenate([data["joint_position"], data["gripper_position"]], axis=-1)[keep].astype(np.float32)
    action = data["action"][keep].astype(np.float32)
    images = {camera: episode.read_video(camera)[keep] for camera in cameras}
    return images, state, action


def export_lerobot(
    sources: list[tuple[Path, int]],
    out: Path,
    cameras: list[str] | None = None,
    outcomes: tuple[str, ...] = ("success",),
    repo_id: str | None = None,
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes: list[tuple[Path, int]] = []
    for source, repeat in sources:
        for outcome in outcomes:
            episodes += [(path, repeat) for path in iter_episodes(source, outcome)]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise SystemExit(f"no {'/'.join(outcomes)} episodes found in {[str(s) for s, _ in sources]}")

    first = load_episode(episodes[0][0])
    cameras = cameras or [c for c in first.meta.cameras if not c.endswith("_right")]
    height, width = first.read_video(cameras[0]).shape[1:3]
    fps = int(round(first.meta.fps))
    features = {
        **{
            f"observation.images.{c}": {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
            for c in cameras
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    }
    out = Path(out)
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    dataset = LeRobotDataset.create(
        repo_id=repo_id or f"local/{out.name}",
        fps=fps,
        features=features,
        root=out,
        robot_type="franka_panda",
        use_videos=True,
    )

    written, frames, skipped = [], 0, Counter()
    for path, repeat in episodes:
        episode = load_episode(path)
        if any(c not in episode.meta.cameras for c in cameras):
            skipped["missing_camera"] += 1
            continue
        images, state, action = episode_arrays(episode, cameras)
        if not (np.isfinite(state).all() and np.isfinite(action).all()):
            skipped["invalid_numerics"] += 1
            continue
        for _ in range(repeat):
            for t in range(len(action)):
                frame = {f"observation.images.{c}": images[c][t] for c in cameras}
                frame.update({"observation.state": state[t], "action": action[t], "task": episode.meta.instruction})
                dataset.add_frame(frame)
            dataset.save_episode()
            frames += len(action)
        written.append({"episode": episode.meta.episode_id, "path": str(path), "repeat": repeat,
                        "steps": int(len(action)), "task": episode.meta.extra.get("task", {}).get("task"),
                        "instruction": episode.meta.instruction, "outcome": episode.meta.outcome})
    dataset.finalize()

    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": [{"path": str(s), "repeat": r} for s, r in sources],
        "outcomes": list(outcomes),
        "cameras": cameras,
        "fps": fps,
        "num_source_episodes": len(written),
        "num_episodes": sum(w["repeat"] for w in written),
        "num_frames": frames,
        "skipped": dict(skipped),
        "per_task": dict(Counter(w["task"] for w in written for _ in range(w["repeat"]))),
        "episodes": written,
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    os.umask(0)  # the container runs as root: let the host user delete what it writes
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", action="append", required=True, metavar="DIR[:REPEAT]", help="Run directory (repeatable).")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--cameras", nargs="+", default=None, help="Default: all recorded cameras except *_right.")
    parser.add_argument("--outcomes", nargs="+", default=["success"], choices=["success", "failure"])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    info = export_lerobot(
        [parse_source(s) for s in args.source], Path(args.out), args.cameras, tuple(args.outcomes), args.repo_id,
        args.overwrite, args.max_episodes,
    )
    print(
        f"[export] {info['num_episodes']} episodes ({info['num_source_episodes']} distinct), {info['num_frames']} frames, "
        f"cameras {info['cameras']}, per task {info['per_task']}, skipped {info['skipped']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
