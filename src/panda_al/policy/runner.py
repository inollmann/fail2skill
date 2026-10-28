"""Chunked execution of a served policy for one env or for many envs at once, with reconnection."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence

import numpy as np
from websockets.exceptions import WebSocketException

from .adapters import Observation, PolicyAdapter
from .websocket_policy import PolicyClient

logger = logging.getLogger(__name__)


class PolicyUnavailable(RuntimeError):
    """The policy server could not be reached again within the allowed attempts."""


class _Connection:
    """A policy server connection that is re-established when it drops.

    ``call(fn)`` runs ``fn(client)`` and, if the connection is lost, reconnects (``reconnect_attempts`` times, 2 s
    apart) and retries before raising :class:`PolicyUnavailable`. Errors reported by the server itself (a bug, a bad
    request) are not retried.
    """

    def __init__(self, host: str, port: int, reconnect_attempts: int = 0):
        self.host, self.port, self.reconnect_attempts = host, port, reconnect_attempts
        self.client: PolicyClient | None = None

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def connect(self):
        """(Re)connect; the server resets the policy on every connection."""
        self.close()
        # PolicyClient itself retries the connection for ``connect_timeout`` seconds
        try:
            self.client = PolicyClient(self.host, self.port, connect_timeout=2.0 * max(1, self.reconnect_attempts))
        except ConnectionError as err:
            raise PolicyUnavailable(f"no policy server at {self.uri}") from err

    def call(self, fn: Callable[[PolicyClient], dict]) -> dict:
        for attempt in range(self.reconnect_attempts + 1):
            try:
                return fn(self.client)
            except (OSError, WebSocketException) as err:
                if attempt == self.reconnect_attempts:
                    raise PolicyUnavailable(f"lost the policy server at {self.uri}") from err
                logger.warning("policy connection lost (%r), reconnecting (%d)", err, attempt + 1)
                time.sleep(2.0)
                try:
                    self.connect()
                except PolicyUnavailable:
                    continue
        raise PolicyUnavailable(f"lost the policy server at {self.uri}")  # pragma: no cover

    def metadata(self) -> dict:
        """The server's metadata as plain JSON (it is stored in ``meta.json``)."""
        return json.loads(json.dumps(self.client.metadata, default=str))

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # the connection may be gone already
                pass
            self.client = None


class PolicyRunner:
    """Query the policy, execute ``adapter.open_loop_horizon`` actions of the chunk, query again (one env).

    One connection per episode (the server resets stateful policies on connect).
    """

    def __init__(self, adapter: PolicyAdapter, host: str, port: int, reconnect_attempts: int = 0):
        self.adapter = adapter
        self._connection = _Connection(host, port, reconnect_attempts)
        self.queue: list[np.ndarray] = []

    @property
    def uri(self) -> str:
        return self._connection.uri

    @property
    def client(self) -> PolicyClient | None:
        return self._connection.client

    def start_episode(self) -> dict:
        """(Re)connect, which resets the policy. Returns the server's metadata as plain JSON."""
        self._connection.connect()
        self.queue = []
        return self._connection.metadata()

    def act(self, obs: Observation) -> np.ndarray:
        if not self.queue:
            response = self._connection.call(lambda client: client.infer(self.adapter.request(obs)))
            chunk = np.asarray(response["actions"])
            self.queue = list(chunk.reshape(-1, chunk.shape[-1])[: self.adapter.open_loop_horizon])
        return self.adapter.to_env_action(self.queue.pop(0), obs)

    def close(self):
        self._connection.close()


class BatchedPolicyRunner:
    """One connection for all envs: every step, the envs whose chunk is used up are queried in a single request.

    Needs a server with batching (``metadata["batched"]``, see :func:`panda_al.policy.serve_policy`) and a policy
    that keeps no state between observations. Envs finish episodes at different times, so the batch size varies
    (at most the number of envs, split further if the server announces ``max_batch``).
    """

    def __init__(self, adapter: PolicyAdapter, host: str, port: int, num_envs: int, reconnect_attempts: int = 0):
        self.adapter = adapter
        self.num_envs = num_envs
        self._connection = _Connection(host, port, reconnect_attempts)
        self.queues: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self.requests = 0
        self.items = 0
        self.server_ms = 0.0
        """Totals since :meth:`connect`: requests sent, observations in them, model time reported by the server."""

    @property
    def uri(self) -> str:
        return self._connection.uri

    @property
    def client(self) -> PolicyClient | None:
        return self._connection.client

    def connect(self) -> dict:
        """Connect (this resets the policy) and return the server's metadata as plain JSON."""
        self._connection.connect()
        if not self._connection.client.metadata.get("batched"):
            raise PolicyUnavailable(f"the policy server at {self.uri} does not support batched requests")
        return self._connection.metadata()

    def start_episodes(self, env_ids: Sequence[int]):
        """Forget the unused actions of the envs that start a new episode."""
        for env_id in env_ids:
            self.queues[env_id] = []

    def needs_query(self) -> list[int]:
        """Envs that need a new chunk at this step."""
        return [i for i in range(self.num_envs) if not self.queues[i]]

    def act(self, observations: Sequence[Observation]) -> list[np.ndarray]:
        """Env actions for this step. ``observations[i]`` needs images only for the envs in :meth:`needs_query`."""
        need = self.needs_query()
        limit = int(self._connection.client.metadata.get("max_batch") or len(need) or 1)
        for start in range(0, len(need), limit):
            part = need[start : start + limit]
            requests = [self.adapter.request(observations[i]) for i in part]
            response = self._connection.call(lambda client, r=requests: client.infer_batch(r))
            chunks = np.asarray(response["actions"])
            self.requests += 1
            self.items += len(part)
            self.server_ms += float(response.get("server_timing", {}).get("infer_ms", 0.0))
            for env_id, chunk in zip(part, chunks, strict=True):
                self.queues[env_id] = list(chunk.reshape(-1, chunk.shape[-1])[: self.adapter.open_loop_horizon])
        return [self.adapter.to_env_action(self.queues[i].pop(0), observations[i]) for i in range(self.num_envs)]

    def close(self):
        self._connection.close()
