"""Websocket policy client and server (openpi protocol, see :mod:`panda_al.policy`)."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

from . import msgpack_numpy
from .adapters import decode_images

logger = logging.getLogger(__name__)


class Policy(Protocol):
    """What :func:`serve_policy` serves: ``infer`` maps one observation dict to ``{"actions": (H, D), ...}``.

    Optionally ``infer_batch`` maps a list of observation dicts to ``{"actions": (B, H, D), ...}`` in one model call
    (otherwise the server calls ``infer`` once per observation). A batch must give each observation the same result as
    a single call, i.e. the policy must not keep state between observations (chunk-based policies don't).
    """

    metadata: dict[str, Any]

    def infer(self, obs: dict) -> dict: ...

    def reset(self) -> None: ...


class PolicyClient:
    """Blocking client for a policy server (ours or openpi's ``serve_policy.py``)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, connect_timeout: float = 300.0):
        self.uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self._conn = connect(self.uri, max_size=None, compression=None, open_timeout=10)
                break
            except (OSError, TimeoutError) as err:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"No policy server at {self.uri}") from err
                logger.info("waiting for policy server at %s ...", self.uri)
                time.sleep(2.0)
        self.metadata: dict = msgpack_numpy.unpackb(self._conn.recv())

    def infer(self, obs: dict) -> dict:
        self._conn.send(self._packer.pack(obs))
        response = self._conn.recv()
        if isinstance(response, str):
            # the server sends its traceback as text before closing the connection
            raise RuntimeError(f"Error in policy server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_batch(self, observations: list[dict]) -> dict:
        """Several observations in one request: ``{"actions": (B, H, D), "server_timing": ...}``.

        Needs a server that announces ``batched`` in its metadata (ours do); openpi's own server does not.
        """
        if not self.metadata.get("batched"):
            raise RuntimeError(f"The policy server at {self.uri} does not support batched requests")
        response = self.infer({"batched": True, "observations": list(observations)})
        if len(response["actions"]) != len(observations):
            raise RuntimeError(f"Asked for {len(observations)} actions, got {len(response['actions'])}")
        return response

    def close(self):
        self._conn.close()


def serve_policy(policy: Policy, host: str = "0.0.0.0", port: int = 8000):
    """Serve ``policy`` until interrupted. Each new connection resets the policy (one rollout per client)."""
    packer = msgpack_numpy.Packer()
    model_lock = threading.Lock()  # several clients (e.g. one simulator per GPU) may share the model, one call at a time
    metadata = {**policy.metadata, "batched": True}

    decode_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jpeg")

    def infer(obs: dict) -> dict:
        obs = decode_images(obs, decode_pool)  # JPEG images (adapter.jpeg_quality) back to arrays
        if obs.get("batched"):
            items = obs["observations"]
            if hasattr(policy, "infer_batch"):
                result = policy.infer_batch(items)
            else:  # no native batching: still saves the round trips
                result = {"actions": np.stack([np.asarray(policy.infer(item)["actions"]) for item in items])}
            result["batch_size"] = len(items)
            return result
        return policy.infer(obs)

    def handler(conn: ServerConnection):
        peer = conn.remote_address
        logger.info("client connected: %s", peer)
        with model_lock:
            policy.reset()
        conn.send(packer.pack(metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(conn.recv())
                start = time.monotonic()
                with model_lock:
                    result = infer(obs)
                result["server_timing"] = {"infer_ms": (time.monotonic() - start) * 1000.0}
                conn.send(packer.pack(result))
            except ConnectionClosed:
                logger.info("client disconnected: %s", peer)
                return
            except Exception:
                conn.send(traceback.format_exc())
                conn.close(code=1011, reason="Internal server error")
                raise

    with serve(handler, host, port, max_size=None, compression=None) as server:
        logger.info("serving %s on ws://%s:%d", policy.metadata.get("name", "policy"), host, port)
        server.serve_forever()
