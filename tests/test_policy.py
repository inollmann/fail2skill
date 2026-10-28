"""Tests for the policy deployment framework (no simulator, no ML framework)."""

import socket
import threading

import numpy as np
import pytest

from panda_al.policy import BatchedPolicyRunner, msgpack_numpy
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


# ---------------------------------------------------------------- batching


class _CountingPolicy(HoldPolicy):
    """Hold policy that remembers how it was called."""

    def __init__(self, horizon=4):
        super().__init__(horizon)
        self.single_calls = 0
        self.batch_sizes = []

    def infer(self, obs):
        self.single_calls += 1
        return super().infer(obs)

    def infer_batch(self, observations):
        self.batch_sizes.append(len(observations))
        return {"actions": np.stack([HoldPolicy.infer(self, obs)["actions"] for obs in observations])}


def _serve(policy):
    port = _free_port()
    threading.Thread(target=serve_policy, args=(policy, "127.0.0.1", port), daemon=True).start()
    return PolicyClient("127.0.0.1", port, connect_timeout=10.0)


def _requests(n, adapter=None):
    adapter = adapter or NativeAdapter()
    return [adapter.request(_obs()) for _ in range(n)]


def test_batched_request_is_one_model_call():
    policy = _CountingPolicy()
    client = _serve(policy)
    assert client.metadata["batched"] is True
    response = client.infer_batch(_requests(5))
    assert response["actions"].shape == (5, 4, 8) and response["batch_size"] == 5
    assert policy.batch_sizes == [5] and policy.single_calls == 0
    client.close()


def test_server_batches_policies_without_native_batching():
    class Plain:  # only ``infer``, like a policy written for openpi's server
        metadata = {"name": "plain"}

        def reset(self):
            pass

        def infer(self, obs):
            return {"actions": np.full((3, 8), obs["observation/joint_position"][0], np.float32)}

    client = _serve(Plain())
    obs = _requests(3)
    for i, request in enumerate(obs):
        request["observation/joint_position"] = np.full(7, float(i), np.float32)
    actions = client.infer_batch(obs)["actions"]
    assert actions.shape == (3, 3, 8) and [float(a[0, 0]) for a in actions] == [0.0, 1.0, 2.0]
    client.close()


def test_batched_runner_queries_only_envs_without_actions():
    policy = _CountingPolicy(horizon=6)
    client = _serve(policy)
    adapter = NativeAdapter(open_loop_horizon=3)
    runner = BatchedPolicyRunner(adapter, "127.0.0.1", int(client.uri.rsplit(":", 1)[1]), 4)
    runner.connect()
    observations = [_obs() for _ in range(4)]
    assert runner.needs_query() == [0, 1, 2, 3]
    runner.act(observations)  # first step: everybody asks
    assert policy.batch_sizes == [4] and runner.needs_query() == []
    runner.act(observations)
    runner.act(observations)  # 3 actions per chunk are used up now
    assert policy.batch_sizes == [4] and runner.needs_query() == [0, 1, 2, 3]
    runner.start_episodes([1, 3])  # these envs start a new episode: their leftover actions are dropped
    runner.queues[0] = [np.zeros(8)]
    runner.queues[2] = [np.zeros(8)]
    assert runner.needs_query() == [1, 3]
    runner.act(observations)
    assert policy.batch_sizes == [4, 2] and runner.requests == 2 and runner.items == 6
    runner.close()
    client.close()


def test_batched_runner_splits_at_max_batch():
    policy = _CountingPolicy()
    policy.metadata = {"name": "small", "max_batch": 2}
    client = _serve(policy)
    runner = BatchedPolicyRunner(NativeAdapter(), "127.0.0.1", int(client.uri.rsplit(":", 1)[1]), 5)
    runner.connect()
    runner.act([_obs() for _ in range(5)])
    assert policy.batch_sizes == [2, 2, 1]
    runner.close()
    client.close()


def test_openpi_wrapper_batches_like_single_calls():
    """Against a stand-in with the attributes of ``openpi.policies.policy.Policy`` that the wrapper relies on."""
    from panda_al.policy.serve import OpenPiPolicy

    class FakePolicy:
        metadata = {"name": "fake"}
        _is_pytorch_model = True
        _pytorch_device = "cpu"
        _sample_kwargs = {}
        seen_batches = []

        def _input_transform(self, obs):  # openpi: renames / normalises keys of one observation
            return {"state": obs["observation/joint_position"] * 2.0, "image": {"wrist": obs["wrist"]}}

        def _output_transform(self, out):  # openpi: de-normalise, cut to the robot's dimensions
            return {"actions": out["actions"][:, :3] + out["state"][:3]}

        def _sample_actions(self, device, observation, **kwargs):
            FakePolicy.seen_batches.append(observation["state"].shape[0])
            state = observation["state"]
            return state[:, None, :].repeat(1, 4, 1)  # (B, H=4, D=7), depends on each observation's state

    observation_cls = type("Obs", (), {"from_dict": staticmethod(lambda d: d)})
    policy = OpenPiPolicy(FakePolicy(), buckets=(1, 4, 8), observation_cls=observation_cls)
    observations = [
        {"observation/joint_position": np.full(7, float(i), np.float32), "wrist": np.zeros((2, 2, 3), np.uint8)}
        for i in range(3)
    ]
    batched = policy.infer_batch(observations)["actions"]
    assert batched.shape == (3, 4, 3)
    assert FakePolicy.seen_batches == [4]  # padded from 3 to the next bucket
    single = [policy.infer(obs)["actions"] for obs in observations]
    for i in range(3):
        np.testing.assert_allclose(batched[i], single[i])
        assert float(batched[i][0, 0]) == 4.0 * i  # state * 2, added once from state and once from the actions


def test_jpeg_images_round_trip_through_the_server():
    seen = {}

    class Spy(HoldPolicy):
        def infer_batch(self, observations):
            seen["image"] = observations[0]["observation/images/wrist"]
            seen["raw"] = observations[0]["observation/state"]
            return super().infer_batch(observations)

    client = _serve(Spy())
    adapter = NativeAdapter(jpeg_quality=90)
    obs = _obs()
    obs.images["wrist"] = np.tile(np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None], (360, 1, 3))
    request = adapter.request(obs)
    assert set(request["observation/images/wrist"]) == {"jpeg"}  # on the wire: bytes
    raw_size = obs.images["wrist"].nbytes
    assert len(request["observation/images/wrist"]["jpeg"]) < raw_size / 10
    client.infer_batch([request, request])
    assert seen["image"].shape == (360, 640, 3) and seen["image"].dtype == np.uint8
    assert np.abs(seen["image"].astype(int) - obs.images["wrist"]).mean() < 3  # lossy, but close
    assert seen["raw"].shape == (8,)  # non-image values are untouched
    client.close()
