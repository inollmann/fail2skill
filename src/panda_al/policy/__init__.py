"""Deploying robot policies (SmolVLA, pi0.5, ...) in the simulator.

A policy runs as its own process (its own container / Python environment / machine) behind a websocket
server that speaks the `openpi <https://github.com/Physical-Intelligence/openpi>`_ protocol: on connect the
server sends a metadata dict, then every request is a msgpack-numpy observation dict and every reply an
action dict with an ``"actions"`` array of shape ``(H, D)`` (an action chunk). Because of that:

* openpi policies (pi0.5, pi0-FAST) are served with openpi's own ``scripts/serve_policy.py``;
* LeRobot policies (SmolVLA, ACT, ...) are served with :mod:`panda_al.policy.serve`;
* the simulator side (``scripts/run_policy.py``) does not depend on any policy framework.

What goes over the wire differs between policy families. An :mod:`adapter <panda_al.policy.adapters>` turns
the simulator's observation into the request a policy expects and the policy's actions into the env's 8-D
absolute joint position actions.

This package only needs ``numpy``, ``msgpack`` and ``websockets`` (no Isaac Lab, no torch).
"""

from .adapters import ADAPTERS, Observation, PolicyAdapter
from .runner import BatchedPolicyRunner, PolicyRunner, PolicyUnavailable
from .websocket_policy import PolicyClient, serve_policy

__all__ = [
    "ADAPTERS",
    "BatchedPolicyRunner",
    "Observation",
    "PolicyAdapter",
    "PolicyClient",
    "PolicyRunner",
    "PolicyUnavailable",
    "serve_policy",
]
