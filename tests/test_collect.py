"""Collection campaign logic: campaign files, ledger/quotas, failure tags, deferred episode writing (no simulator)."""

import json

import numpy as np
import pytest

pytest.importorskip("isaaclab.controllers")  # panda_al.scripted (campaign modes) imports Isaac Lab

from panda_al.collect.campaign import load_campaign, parse_campaign  # noqa: E402
from panda_al.collect.ledger import Ledger  # noqa: E402
from panda_al.collect.tags import failure_tags  # noqa: E402
from panda_al.episode_io import (  # noqa: E402
    EpisodeMeta,
    EpisodeWriter,
    deferred_buffer_bytes,
    load_episode,
    wait_for_pending_episodes,
)

# ---------------------------------------------------------------- campaign


def test_example_campaigns_are_valid():
    for name in ("example", "dryrun_expert"):
        campaign = load_campaign(f"configs/campaigns/{name}.yaml")
        assert campaign.name and campaign.num_envs >= 1
    example = load_campaign("configs/campaigns/example.yaml")
    assert example.save.outcomes == ["failure"] and example.save.quota_per_task == 200
    assert example.obstacle_probability == 0.1 and example.tasks["stack"] == 1.0


def test_policy_batch_setting_accepts_yaml_booleans():
    assert parse_campaign({"name": "x"}).policy.batch == "auto"
    assert parse_campaign({"name": "x", "policy": {"batch": False}}).policy.batch == "false"
    assert parse_campaign({"name": "x", "policy": {"batch": True, "jpeg_quality": 90}}).policy.batch == "true"
    with pytest.raises(ValueError, match="batch"):
        parse_campaign({"name": "x", "policy": {"batch": "sometimes"}})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "x", "driver": "magic"}, "driver"),
        ({"name": "x", "policy": {"adapter": "nope"}}, "adapter"),
        ({"name": "x", "save": {"outcomes": []}}, "outcomes"),
        ({"name": "x", "save": {"quota": {"per_task": 1}, "typo": 1}}, "unknown keys"),
        ({"name": "x", "perturbations": {"rain": {}}}, "perturbations"),
        ({"name": "x", "perturbations": {"obstacle": {"probability": 2}}}, "probabilities"),
        ({"name": "x", "bogus": 1}, "unknown keys"),
        ({"driver": "policy"}, "name"),
    ],
)
def test_invalid_campaigns_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        parse_campaign(data)


# ---------------------------------------------------------------- ledger


def test_ledger_quotas_drop_tags_and_resume(tmp_path):
    ledger = Ledger(tmp_path, ["a", "b"], quota_per_task=2, quota_per_tag=2, drop_tags=["stalled_at_start"])
    assert ledger.decide("a", ["stalled_at_start"]) == (False, "dropped_tags")
    assert ledger.decide("a", ["stalled_at_start", "timeout"]) == (True, "kept")  # another tag: informative
    assert ledger.decide("a", []) == (True, "kept")

    def finish(task, tags, outcome="failure", kept=True):
        ledger.record({"task": task, "outcome": outcome, "tags": tags, "episode_id": "e"}, kept, "r")()

    finish("a", ["collision"])
    finish("a", ["collision"])
    finish("a", [], "success", kept=False)
    assert ledger.decide("a", ["no_grasp"]) == (False, "task_quota_full")
    assert ledger.decide("b", ["collision"]) == (False, "tag_quota_full")
    assert ledger.decide("b", ["collision", "timeout"]) == (True, "kept")
    assert not ledger.quotas_met()
    finish("b", ["timeout"])
    finish("b", ["no_grasp"])
    assert ledger.quotas_met()
    assert ledger.success_rates()["a"] == pytest.approx(1 / 3)

    # a restarted run continues with the same counts
    again = Ledger(tmp_path, ["a", "b"], quota_per_task=2, quota_per_tag=2)
    assert again.quotas_met() and again.total_finished == 5 and again.total_saved == 4
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 4  # successes that were dropped are not in it
    assert json.loads((tmp_path / "stats.jsonl").read_text().splitlines()[2])["kept"] is False


def test_ledger_writes_kept_episodes_only_when_persisted(tmp_path):
    ledger = Ledger(tmp_path, ["a"], quota_per_task=1)
    persist = ledger.record({"task": "a", "outcome": "failure", "tags": ["timeout"], "episode_id": "e1"}, True, "kept")
    assert ledger.decide("a", ["timeout"]) == (False, "task_quota_full")  # counts at once, so quotas cannot overshoot
    assert not ledger.stats_path.exists() and not ledger.manifest_path.exists()  # but a crash now leaves no line
    persist()
    assert len(ledger.manifest_path.read_text().splitlines()) == 1
    assert Ledger(tmp_path, ["a"]).saved_per_task["a"] == 1


# ---------------------------------------------------------------- tags

NAMES = ["obj_blue_cube", "obj_red_cube"]
STACK = {"pick": "red_cube", "place": "blue_cube", "goal": {"type": "stack"}}


def _episode(pick_z, pick_xy=None, tcp=None):
    t = len(pick_z)
    poses = np.zeros((t, 2, 7))
    poses[:, 0, :3] = [0.5, 0.0, 0.0234]  # blue cube (place)
    poses[:, 1, 2] = pick_z
    poses[:, 1, :2] = pick_xy if pick_xy is not None else [0.45, 0.1]
    tcp_path = np.zeros((t, 6))
    tcp_path[:, 0] = np.linspace(0.3, 0.3 + (tcp if tcp is not None else 0.4), t)
    return {"cartesian_position": tcp_path, "object_pose": poses}


def test_failure_tags():
    rest = 0.0234
    # never moved
    assert failure_tags(_episode([rest] * 20, tcp=0.01), NAMES, STACK, ["time_out"]) == ["stalled_at_start", "timeout"]
    # moved but never lifted the cube
    assert failure_tags(_episode([rest] * 20), NAMES, STACK, ["time_out"]) == ["no_grasp", "timeout"]
    # lifted, then lying on the table away from the goal
    lifted = [rest] * 5 + [0.1] * 10 + [rest] * 5
    assert failure_tags(_episode(lifted), NAMES, STACK, ["time_out"]) == ["dropped", "timeout"]
    # lifted and still in the air at the end
    assert failure_tags(_episode([rest] * 5 + [0.1] * 15), NAMES, STACK, ["time_out"]) == ["misplaced", "timeout"]
    # obstacle hit and object knocked off
    tags = failure_tags(_episode([rest] * 20), NAMES, STACK, ["obstacle_collision", "object_dropped"])
    assert tags == ["no_grasp", "collision", "knocked_off"]
    # beside goal: object lifted, put down at the target spot but at the wrong height is still not "dropped"
    beside = {**STACK, "goal": {"type": "beside", "direction": "left", "distance": 0.1}}
    episode = _episode(lifted, pick_xy=[0.5, 0.1])  # exactly the target spot
    assert failure_tags(episode, NAMES, beside, ["time_out"]) == ["misplaced", "timeout"]
    assert failure_tags(_episode([rest] * 20, tcp=0.0), NAMES, STACK, []) == ["stalled_at_start"]


# ---------------------------------------------------------------- deferred writer


def _writer(tmp_path, defer=True, steps=4):
    meta = EpisodeMeta(
        episode_id="ep", task="t", instruction="i", source="test", fps=15.0, cameras=["cam"], object_names=[],
        arm_joint_names=[],
    )
    writer = EpisodeWriter(tmp_path, meta, defer=defer)
    writer.set_initial_state({"x": np.arange(3)})
    for t in range(steps):
        frame = np.full((36, 64, 3), t, np.uint8)
        writer.append({"cam": frame}, {"action": np.full(8, t, np.float32)})
    return writer


def test_deferred_writer_discard_costs_nothing(tmp_path):
    before = deferred_buffer_bytes()
    writer = _writer(tmp_path)
    assert deferred_buffer_bytes() == before + 4 * 36 * 64 * 3
    assert writer.finalize("discard") is None
    assert deferred_buffer_bytes() == before
    assert not list(tmp_path.rglob("*"))  # nothing was ever written, not even a directory


def test_deferred_writer_saves_in_background(tmp_path):
    before = deferred_buffer_bytes()
    writer = _writer(tmp_path)
    path = writer.finalize("failure", ["time_out"])
    assert wait_for_pending_episodes() == []
    assert path == tmp_path / "failure" / "ep" and (path / "videos" / "cam.mp4").exists()
    episode = load_episode(path)
    assert episode.meta.outcome == "failure" and episode.meta.termination_terms == ["time_out"]
    assert episode.data["action"].shape == (4, 8)
    assert episode.read_video("cam").shape == (4, 36, 64, 3)
    assert episode.initial_state["x"].tolist() == [0, 1, 2]
    assert deferred_buffer_bytes() == before
    assert not (tmp_path / "in_progress" / "ep").exists()


def test_deferred_writer_reports_when_saved(tmp_path):
    seen = []
    writer = _writer(tmp_path)
    writer.on_saved = lambda path: seen.append((path, (path / "meta.json").exists()))
    path = writer.finalize("failure")
    wait_for_pending_episodes()
    assert seen == [(path, True)]  # called after the episode is complete and in place


def test_deferred_frames_are_copied(tmp_path):
    meta = EpisodeMeta(
        episode_id="ep2", task="t", instruction="i", source="test", fps=15.0, cameras=["cam"], object_names=[],
        arm_joint_names=[],
    )
    writer = EpisodeWriter(tmp_path, meta, defer=True)
    batch = np.zeros((2, 36, 64, 3), np.uint8)
    writer.append({"cam": batch[1]}, {"action": np.zeros(8)})
    batch[:] = 255  # the sim reuses its buffers: the writer must not see this
    path = writer.finalize("failure")
    wait_for_pending_episodes()
    assert load_episode(path).read_video("cam").max() < 20  # (lossy codec)
