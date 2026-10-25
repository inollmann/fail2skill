"""Serve a policy over websocket in this project's native format (see :class:`~panda_al.policy.adapters.NativeAdapter`).

.. code-block:: bash

    # smoke test: holds the current joint positions (no ML dependencies)
    python -m panda_al.policy.serve hold

    # any LeRobot policy (SmolVLA, ACT, diffusion, ...) from a local dir or the Hugging Face hub
    python -m panda_al.policy.serve lerobot --checkpoint <dir or repo id> \\
        --rename observation.images.wrist=observation.images.camera1

openpi policies (pi0.5) are served with openpi's own ``scripts/serve_policy.py`` instead (see README).
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np


class HoldPolicy:
    """Returns a chunk that holds the current joint positions with the gripper open."""

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.metadata = {"name": "hold", "action_horizon": horizon}

    def reset(self):
        pass

    def infer(self, obs: dict) -> dict:
        action = np.append(obs["observation/joint_position"], 0.0).astype(np.float32)
        return {"actions": np.tile(action, (self.horizon, 1))}


class LeRobotPolicy:
    """Wraps a LeRobot policy (needs ``lerobot``; follows ``lerobot.async_inference.policy_server`` of lerobot 0.6).

    Native observations become LeRobot features: ``observation/images/<cam>`` -> ``observation.images.<cam>``,
    ``observation/state`` -> ``observation.state``, ``prompt`` -> ``task``. ``rename_map`` renames these to the
    policy's feature names where they differ (LeRobot's ``rename_observations_processor``). The returned chunk
    must be in the env's action format, which holds for policies trained on this project's episodes.
    """

    def __init__(self, checkpoint: str, rename_map: dict[str, str] | None = None, device: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy_type = PreTrainedConfig.from_pretrained(checkpoint).type
        self.policy = get_policy_class(policy_type).from_pretrained(checkpoint)
        self.policy.to(self.device)
        device_override = {"device": self.device}
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": rename_map or {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        self.metadata = {"name": f"lerobot/{policy_type}", "checkpoint": checkpoint}

    def reset(self):
        self.policy.reset()
        self.preprocess.reset()
        self.postprocess.reset()

    def infer(self, obs: dict) -> dict:
        from lerobot.policies.utils import prepare_observation_for_inference

        torch = self._torch
        features = {
            key.replace("observation/images/", "observation.images."): np.ascontiguousarray(value)
            for key, value in obs.items()
            if key.startswith("observation/images/")
        }
        features["observation.state"] = np.asarray(obs["observation/state"], dtype=np.float32)
        with torch.inference_mode():
            batch = prepare_observation_for_inference(features, torch.device(self.device), obs.get("prompt", ""))
            chunk = self.policy.predict_action_chunk(self.preprocess(batch))
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(1)
            # the postprocessor works on (B, action_dim): one action at a time
            actions = torch.stack([self.postprocess(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)[0]
        return {"actions": actions.float().cpu().numpy()}


def main():
    from .websocket_policy import serve_policy

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("policy", choices=["hold", "lerobot"])
    parser.add_argument("--checkpoint", help="lerobot: local directory or Hugging Face repo id.")
    parser.add_argument(
        "--rename", action="append", default=[], metavar="OURS=POLICY", help="lerobot: rename an observation feature."
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AL_POLICY_PORT", "8000")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if args.policy == "hold":
        policy = HoldPolicy()
    else:
        if not args.checkpoint:
            parser.error("lerobot needs --checkpoint")
        policy = LeRobotPolicy(args.checkpoint, dict(kv.split("=", 1) for kv in args.rename), args.device)
    serve_policy(policy, args.host, args.port)


if __name__ == "__main__":
    main()
