"""Fine-tune a LeRobot policy (default: SmolVLA base) on a dataset exported by :mod:`panda_al.export.lerobot_export`.

.. code-block:: bash

    ./al.sh train --dataset data/lerobot/demos_v0 --name smolvla_iter0 --steps 20000 --batch_size 16
    ./al.sh train --dataset data/lerobot/demos_v0 --name smolvla_iter0 -- --policy.optimizer_lr=5e-5   # extra lerobot-train flags

A thin wrapper around ``lerobot-train``: it fills in the dataset location, the output directory
(``$AL_DATA/checkpoints/<name>/``, checkpoints every ``--save_freq`` steps: keep the early, mid and final ones as a ladder
of competence), renames our cameras to the ones of ``smolvla_base`` (``camera1..3``) and prints the command that serves
the result. Runs in the policy image on any CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SMOLVLA_CAMERAS = ["camera1", "camera2", "camera3"]


def camera_rename_map(cameras: list[str]) -> dict[str, str]:
    """Our camera features -> the ones the base model was trained with."""
    if len(cameras) > len(SMOLVLA_CAMERAS):
        raise SystemExit(f"smolvla_base has {len(SMOLVLA_CAMERAS)} camera inputs, the dataset has {cameras}")
    return {f"observation.images.{ours}": f"observation.images.{theirs}" for ours, theirs in zip(cameras, SMOLVLA_CAMERAS)}


def build_command(args: argparse.Namespace, extra: list[str]) -> tuple[list[str], dict[str, str]]:
    dataset = Path(args.dataset)
    info = json.loads((dataset / "export_info.json").read_text()) if (dataset / "export_info.json").exists() else {}
    cameras = info.get("cameras") or ["exterior_1", "exterior_2", "wrist"]
    rename = {} if args.no_rename else camera_rename_map(cameras)
    output = Path(args.output_root) / args.name
    command = [
        "lerobot-train",
        f"--policy.path={args.base}",
        "--policy.push_to_hub=false",
        f"--policy.device={args.device}",
        f"--dataset.repo_id=local/{dataset.name}",
        f"--dataset.root={dataset}",
        f"--output_dir={output}",
        f"--job_name={args.name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        "--wandb.enable=false",
        f"--seed={args.seed}",
    ]
    if rename:
        command.append(f"--rename_map={json.dumps(rename)}")
    return command + extra, rename


def serve_hint(output: Path, rename: dict[str, str]) -> str:
    flags = " ".join(f"--rename {ours}={theirs}" for ours, theirs in rename.items())
    return f"./al.sh serve lerobot --checkpoint {output}/checkpoints/last/pretrained_model {flags}"


def main():
    os.umask(0)  # the container runs as root: let the host user delete what it writes
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="Dataset directory (output of export-lerobot).")
    parser.add_argument("--name", required=True, help="Run name: checkpoints go to <output_root>/<name>.")
    parser.add_argument("--base", default="lerobot/smolvla_base", help="Policy to start from (hub id or directory).")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default="/workspace/data/checkpoints")
    parser.add_argument("--no_rename", action="store_true", help="Keep our camera names (policies trained from scratch).")
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != "--"]
    command, rename = build_command(args, extra)
    print("[train] " + " ".join(command), flush=True)
    code = subprocess.run(command).returncode
    output = Path(args.output_root) / args.name
    if code == 0:
        print(f"[train] done. Serve the final checkpoint with:\n  {serve_hint(output, rename)}")
    sys.exit(code)


if __name__ == "__main__":
    main()
