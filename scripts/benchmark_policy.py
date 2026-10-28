"""Latency of a policy server (milestone M0 of docs/automated_collection.md). No simulator needed.

.. code-block:: bash

    ./al.sh serve lerobot --checkpoint ...            # or openpi's serve_policy.py, on this or another machine
    python scripts/benchmark_policy.py --adapter droid --host <server> --port 8000

Sends synthetic observations in the adapter's format (random images, joint positions) and reports round-trip latency
per request, which includes the network. Run it from this machine against the Jetson Thor to see what the sim sees.
With ``--batch_sizes 1 4 16`` it also measures batched requests (one model call for many envs) against a server that
supports them (ours do), which shows what batching is worth for this model on this GPU.
"""

import argparse
import json
import time

import numpy as np

from panda_al.policy import ADAPTERS, Observation, PolicyClient

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="native")
parser.add_argument("--requests", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5, help="Requests before measuring (JIT, cache warm-up).")
parser.add_argument("--batch_sizes", type=int, nargs="*", default=[], help="Also time batched requests of these sizes.")
parser.add_argument("--jpeg_quality", type=int, default=None, help="Send images as JPEG of this quality.")
parser.add_argument("--state_dim", type=int, default=None, help="Cut observation/state to this size (e.g. 6 for SmolVLA base).")
args = parser.parse_args()

rng = np.random.default_rng(0)
adapter = ADAPTERS[args.adapter]()
adapter.jpeg_quality = args.jpeg_quality
obs = Observation(
    images={name: rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for name in ["exterior_1", "exterior_2", "wrist"]},
    joint_position=rng.uniform(-1, 1, 7),
    gripper_position=np.array([0.0]),
    instruction="put the red cube on top of the blue cube",
)
request = adapter.request(obs)
if args.state_dim is not None and "observation/state" in request:
    request["observation/state"] = request["observation/state"][: args.state_dim]
payload_mb = sum(
    v.nbytes if isinstance(v, np.ndarray) else len(v.get('jpeg', b'')) if isinstance(v, dict) else 0 for v in request.values()
) / 2**20

client = PolicyClient(args.host, args.port, connect_timeout=30.0)
latencies = []
for i in range(args.warmup + args.requests):
    start = time.perf_counter()
    response = client.infer(request)
    elapsed = (time.perf_counter() - start) * 1000
    if i >= args.warmup:
        latencies.append(elapsed)
client.close()

lat = np.array(latencies)
result = {
    "server": client.metadata.get("name"),
    "adapter": args.adapter,
    "request_mb": round(payload_mb, 2),
    "chunk_shape": list(np.asarray(response["actions"]).shape),
    "latency_ms": {"mean": round(lat.mean(), 1), "p50": round(np.percentile(lat, 50), 1), "p95": round(np.percentile(lat, 95), 1)},
    "server_infer_ms": round(float(response.get("server_timing", {}).get("infer_ms", float("nan"))), 1),
    "requests_per_s": round(1000 / lat.mean(), 1),
}
print(json.dumps(result, indent=2))

if args.batch_sizes:
    client = PolicyClient(args.host, args.port, connect_timeout=30.0)
    rows = {}
    for size in args.batch_sizes:
        batch = [request] * size
        times = []
        for i in range(args.warmup + args.requests):
            start = time.perf_counter()
            response = client.infer_batch(batch)
            elapsed = (time.perf_counter() - start) * 1000
            if i >= args.warmup:
                times.append(elapsed)
        mean = float(np.mean(times))
        rows[size] = {
            "latency_ms": round(mean, 1),
            "ms_per_env": round(mean / size, 1),
            "server_ms": round(float(response["server_timing"]["infer_ms"]), 1),
            "request_mb": round(payload_mb * size, 2),
        }
    client.close()
    print(json.dumps({"batched": rows}, indent=2))
