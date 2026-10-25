"""Tests for the policy deployment framework (no simulator, no ML framework)."""

import socket
import threading

import numpy as np
import pytest

from panda_al.policy import msgpack_numpy
from panda_al.policy.adapters import DroidAdapter, NativeAdapter, Observation, resize_with_pad
from panda_al.policy.serve import HoldPolicy
from panda_al.policy.websocket_policy import PolicyClient, serve_policy


def _obs() -> Observation:
    return Observation(
        images={name: np.full((360, 640, 3), i, np.uint8) for i, name in enumerate(["exterior_1", "exterior_2", "wrist"])},
        joint_position=np.linspace(-1.0, 1.0, 7),
        gripper_position=np.array([0.25]),
        instruction="stack the cubes",
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_msgpack_numpy_roundtrip():
    data = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.float64(1.5), "s": "text"}
    out = msgpack_numpy.unpackb(msgpack_numpy.packb(data))
    np.testing.assert_array_equal(out["a"], data["a"])
    assert out["a"].dtype == np.float32 and out["b"] == 1.5 and out["s"] == "text"


def test_resize_with_pad_keeps_aspect():
    image = np.full((360, 640, 3), 200, np.uint8)
    out = resize_with_pad(image, 224, 224)
    assert out.shape == (224, 224, 3)
    # 16:9 -> 224x126 content, zero bands top and bottom
    assert out[0].max() == 0 and out[-1].max() == 0 and out[112, 112, 0] == 200


def test_native_adapter():
    adapter = NativeAdapter()
    req = adapter.request(_obs())
    assert set(req) == {
        "observation/images/exterior_1",
        "observation/images/exterior_2",
        "observation/images/wrist",
        "observation/joint_position",
        "observation/gripper_position",
        "observation/state",
        "prompt",
    }
    assert req["observation/state"].shape == (8,)
    action = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(adapter.to_env_action(action, _obs()), action)


def test_droid_adapter_velocity_to_position():
    adapter = DroidAdapter()
    obs = _obs()
    req = adapter.request(obs)
    assert req["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert req["observation/wrist_image_left"].shape == (224, 224, 3)
    velocity = np.array([0.5, -2.0, 0, 0, 0, 0, 1.0, 0.7])
    action = adapter.to_env_action(velocity, obs)
    expected = obs.joint_position + np.clip(velocity[:7], -1, 1) * adapter.max_joint_delta
    np.testing.assert_allclose(action[:7], expected, rtol=1e-6)
    assert action[7] == 1.0


def test_server_client_roundtrip():
    port = _free_port()
    threading.Thread(target=serve_policy, args=(HoldPolicy(horizon=4), "127.0.0.1", port), daemon=True).start()
    client = PolicyClient("127.0.0.1", port, connect_timeout=10.0)
    try:
        assert client.metadata["name"] == "hold"
        obs = _obs()
        result = client.infer(NativeAdapter().request(obs))
        assert result["actions"].shape == (4, 8)
        np.testing.assert_allclose(result["actions"][0, :7], obs.joint_position, rtol=1e-6)
        assert "server_timing" in result
    finally:
        client.close()


def test_server_reports_errors():
    port = _free_port()
    threading.Thread(target=serve_policy, args=(HoldPolicy(), "127.0.0.1", port), daemon=True).start()
    client = PolicyClient("127.0.0.1", port, connect_timeout=10.0)
    with pytest.raises(RuntimeError, match="KeyError"):
        client.infer({"prompt": "missing joint positions"})
