"""Episodes -> LeRobotDataset (needs lerobot with the dataset extra: runs in the policy image, ``./al.sh test-policy``)."""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")

from panda_al.episode_io import EpisodeMeta, EpisodeWriter  # noqa: E402
from panda_al.export.lerobot_export import export_lerobot, parse_source  # noqa: E402

H, W, STEPS = 36, 64, 6


def _episode(run_dir, episode_id, outcome, task="stack", prompt="stack it", cameras=("exterior_1", "wrist")):
    meta = EpisodeMeta(
        episode_id=episode_id, task="PandaAL-MultiTask-v0", instruction=prompt, source="test", fps=15.0,
        cameras=list(cameras), object_names=[], arm_joint_names=[], extra={"task": {"task": task}},
    )
    writer = EpisodeWriter(run_dir, meta)
    for t in range(STEPS):
        writer.append(
            {c: np.full((H, W, 3), 40 * i + 10 * t, np.uint8) for i, c in enumerate(cameras)},
            {
                "action": np.full(8, t, np.float32),
                "joint_position": np.full(7, 0.1 * t, np.float32),
                "gripper_position": np.array([float(t % 2)], np.float32),
                "is_last": np.bool_(t == STEPS - 1),
            },
        )
    writer.finalize(outcome)


def test_parse_source():
    assert parse_source("data/episodes/x") == (parse_source("data/episodes/x")[0], 1)
    assert parse_source("data/episodes/x:3")[1] == 3 and str(parse_source("data/episodes/x:3")[0]) == "data/episodes/x"


def test_export_roundtrip(tmp_path):
    run = tmp_path / "run"
    _episode(run, "a", "success", "stack", "stack the red cube")
    _episode(run, "b", "success", "beside_left", "move it left")
    _episode(run, "c", "failure", "stack", "this one failed")
    info = export_lerobot([(run, 1)], tmp_path / "ds", cameras=["exterior_1", "wrist"])
    assert info["num_episodes"] == 2 and info["num_frames"] == 2 * (STEPS - 1)  # the terminal row has no action
    assert info["per_task"] == {"stack": 1, "beside_left": 1}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/ds", root=tmp_path / "ds")
    assert dataset.num_episodes == 2 and len(dataset) == 2 * (STEPS - 1) and dataset.fps == 15
    assert tuple(dataset.features["observation.state"]["shape"]) == (8,)
    first = dataset[0]
    assert first["observation.images.exterior_1"].shape == (3, H, W)
    assert first["task"] == "stack the red cube"
    np.testing.assert_allclose(first["observation.state"].numpy(), [0.0] * 7 + [0.0], atol=1e-6)  # t = 0
    np.testing.assert_allclose(dataset[2]["observation.state"].numpy()[:7], 0.2, atol=1e-6)  # t = 2
    np.testing.assert_allclose(dataset[2]["action"].numpy(), 2.0)
    assert {dataset[i]["task"] for i in range(len(dataset))} == {"stack the red cube", "move it left"}
    exported = json.loads((tmp_path / "ds" / "export_info.json").read_text())
    assert [e["episode"] for e in exported["episodes"]] == ["a", "b"] and all(e["outcome"] == "success" for e in exported["episodes"])


def test_repeat_and_outcome_selection(tmp_path):
    run = tmp_path / "run"
    _episode(run, "a", "success")
    _episode(run, "c", "failure")
    info = export_lerobot([(run, 3)], tmp_path / "ds", cameras=["exterior_1"], outcomes=("success", "failure"))
    assert info["num_source_episodes"] == 2 and info["num_episodes"] == 6
    with pytest.raises(SystemExit, match="exists"):
        export_lerobot([(run, 1)], tmp_path / "ds", cameras=["exterior_1"])
    export_lerobot([(run, 1)], tmp_path / "ds", cameras=["exterior_1"], overwrite=True)
    with pytest.raises(SystemExit, match="no failure"):
        export_lerobot([(tmp_path / "empty", 1)], tmp_path / "ds2", outcomes=("failure",))
