"""Unit tests that run without the simulator (``./al.sh test``)."""

import json
import socket
import time

import numpy as np
import pytest

from panda_al.episode_io import (
    EpisodeMeta,
    EpisodeWriter,
    flatten_state,
    iter_episodes,
    load_episode,
    unflatten_state,
)
from panda_al.sim_bridge import BridgeServer, TrajectoryExecutor


def _meta(episode_id="ep0"):
    return EpisodeMeta(
        episode_id=episode_id,
        task="t",
        instruction="i",
        source="test",
        fps=15.0,
        cameras=["cam"],
        object_names=["cube"],
        arm_joint_names=[f"j{i}" for i in range(7)],
    )


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_episode_roundtrip(tmp_path, outcome):
    writer = EpisodeWriter(tmp_path, _meta())
    writer.set_initial_state({"articulation/robot/joint_position": np.ones((1, 9))})
    for t in range(5):
        frame = np.full((36, 64, 3), 40 * t, dtype=np.uint8)
        writer.append({"cam": frame}, {"action": np.full(8, t, np.float32), "is_last": np.bool_(t == 4)})
    path = writer.finalize(outcome, ["time_out"])

    assert path == tmp_path / outcome / "ep0"
    assert [p.name for p in iter_episodes(tmp_path, outcome)] == ["ep0"]
    episode = load_episode(path)
    assert episode.meta.outcome == outcome and episode.meta.num_steps == 5
    assert episode.meta.termination_terms == ["time_out"]
    np.testing.assert_array_equal(episode.data["action"][:, 0], np.arange(5))
    video = episode.read_video("cam")
    assert video.shape == (5, 36, 64, 3)
    # lossy codec, but the brightness ramp must survive
    assert np.all(np.diff(video.mean(axis=(1, 2, 3))) > 20)


def test_depth_roundtrip(tmp_path):
    meta = EpisodeMeta(
        episode_id="d", task="t", instruction="i", source="test", fps=15.0, cameras=["cam"],
        object_names=[], arm_joint_names=[], depth_cameras=["cam"],
    )
    writer = EpisodeWriter(tmp_path, meta)
    depth = [np.random.randint(0, 15000, (36, 64), dtype=np.uint16) for _ in range(3)]
    for t in range(3):
        writer.append({"cam": np.zeros((36, 64, 3), np.uint8)}, {"action": np.zeros(8)}, {"cam": depth[t]})
    path = writer.finalize("success")
    episode = load_episode(path)
    assert episode.meta.depth_cameras == ["cam"]
    np.testing.assert_array_equal(episode.read_depth("cam"), np.stack(depth))


def test_discard_removes_episode(tmp_path):
    writer = EpisodeWriter(tmp_path, _meta())
    writer.append({"cam": np.zeros((36, 64, 3), np.uint8)}, {"action": np.zeros(8)})
    assert writer.finalize("discard") is None
    assert not any(tmp_path.rglob("data.npz"))


def test_state_flatten_roundtrip():
    state = {
        "articulation": {"robot": {"joint_position": np.zeros((1, 9))}},
        "rigid_object": {"cube": {"root_pose": np.ones((1, 7))}},
    }
    restored = unflatten_state(flatten_state(state))
    np.testing.assert_array_equal(
        restored["rigid_object"]["cube"]["root_pose"], state["rigid_object"]["cube"]["root_pose"]
    )


def test_trajectory_executor_interpolates_and_reorders():
    executor = TrajectoryExecutor(["a", "b"], settle_time=0.1)
    hold = np.array([0.0, 5.0])
    # message lists the joints in reverse order and omits t=0
    executor.start(
        {"id": 1, "joint_names": ["b", "a"], "points": [{"t": 1.0, "positions": [7.0, 2.0]}]}, now=10.0, hold=hold
    )
    q, done = executor.target(10.5)
    np.testing.assert_allclose(q, [1.0, 6.0])
    assert not done
    q, done = executor.target(11.2)
    np.testing.assert_allclose(q, [2.0, 7.0])
    assert done and executor.stop() == 1 and not executor.active


def test_bridge_server_roundtrip():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = BridgeServer("127.0.0.1", port)
    client = socket.create_connection(("127.0.0.1", port))
    try:
        client.sendall(b'{"type": "trajectory", "id": 3}\n{"type": "cancel"')
        messages = []
        for _ in range(50):
            messages += server.poll()
            if messages:
                break
            time.sleep(0.01)
        assert messages == [{"type": "trajectory", "id": 3}]
        server.send({"type": "state", "time": 1.5})
        assert json.loads(client.recv(1024).decode().strip()) == {"type": "state", "time": 1.5}
    finally:
        client.close()
        server.close()
