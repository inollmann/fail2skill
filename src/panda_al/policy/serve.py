"""Serve a policy over websocket in this project's native format (see :class:`~panda_al.policy.adapters.NativeAdapter`).

.. code-block:: bash

    # smoke test: holds the current joint positions (no ML dependencies)
    python -m panda_al.policy.serve hold

    # any LeRobot policy (SmolVLA, ACT, ...) from a local dir or the Hugging Face hub
    python -m panda_al.policy.serve lerobot --checkpoint <dir or repo id> \\
        --rename observation.images.wrist=observation.images.camera1

    # an openpi policy (pi0.5 DROID) with batching; needs openpi installed in the environment running this
    python -m panda_al.policy.serve openpi --config pi05_droid --checkpoint gs://openpi-assets/checkpoints/pi05_droid

All of them answer batched requests (``PolicyClient.infer_batch``): one model call for the observations of many envs.
(openpi's own ``scripts/serve_policy.py`` also works for single requests, but does not batch.)
"""

from __future__ import annotations

import argparse
import logging
import os
import time

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

    def infer_batch(self, observations: list[dict]) -> dict:
        return {"actions": np.stack([self.infer(obs)["actions"] for obs in observations])}


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
        n_obs_steps = getattr(self.policy.config, "n_obs_steps", 1)
        if n_obs_steps != 1:
            # a history of observations is kept per call, which breaks when calls mix observations of different envs
            raise ValueError(f"{policy_type} uses {n_obs_steps} observation steps; batched serving needs exactly 1")
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
        result = self.infer_batch([obs])
        return {**result, "actions": result["actions"][0]}

    def infer_batch(self, observations: list[dict]) -> dict:
        """One model call for all observations. Needs a policy without observation history (``n_obs_steps == 1``)."""
        torch = self._torch
        batch = {}
        for key in observations[0]:
            if key.startswith("observation/images/"):
                images = np.stack([np.ascontiguousarray(obs[key]) for obs in observations])  # (B, H, W, 3) uint8
                tensor = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float().div(255.0)
                batch[key.replace("observation/images/", "observation.images.")] = tensor.contiguous()
        state = np.stack([np.asarray(obs["observation/state"], dtype=np.float32) for obs in observations])
        batch["observation.state"] = torch.from_numpy(state).to(self.device)
        batch["task"] = [obs.get("prompt", "") for obs in observations]
        batch["robot_type"] = [""] * len(observations)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocess(batch))
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(1)
            # the postprocessor works on (B, action_dim): one action at a time
            actions = torch.stack([self.postprocess(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)
        return {"actions": actions.float().cpu().numpy()}


class OpenPiPolicy:
    """An openpi policy (pi0.5, pi0-FAST, ...) with batching. Needs ``openpi`` installed (JAX or PyTorch models).

    Written against ``openpi.policies.policy.Policy`` (March 2026 ``main``): the observations of the batch go through its
    input transforms one by one, are stacked and sampled in one model call, and the outputs go through its output
    transforms one by one. It uses private attributes of that class, so a changed openpi may need an update here.
    The batch is padded to the next size in ``buckets`` so that JAX does not recompile for every batch size.
    """

    def __init__(self, policy, buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64), observation_cls=None):
        self.policy = policy
        self.buckets = tuple(sorted(buckets))
        self.metadata = {"name": "openpi", "max_batch": self.buckets[-1], **dict(policy.metadata)}
        if observation_cls is None:
            from openpi.models import model as openpi_model

            observation_cls = openpi_model.Observation
        self.observation_cls = observation_cls

    @classmethod
    def load(cls, config: str, checkpoint: str, **kwargs) -> OpenPiPolicy:
        try:
            from openpi.policies import policy_config
            from openpi.shared import download
            from openpi.training import config as openpi_config
        except ImportError as err:
            raise SystemExit(
                f"openpi is not installed in this environment ({err}). Run this from an openpi checkout "
                "(uv run, Python 3.11) with this repo's src/ on the path: "
                "PYTHONPATH=<repo>/src uv run python -m panda_al.policy.serve openpi --config ... --checkpoint ..."
            ) from err

        policy = policy_config.create_trained_policy(
            openpi_config.get_config(config), download.maybe_download(checkpoint)
        )
        return cls(policy, **kwargs)

    def reset(self):
        pass

    def infer(self, obs: dict) -> dict:
        result = self.infer_batch([obs])
        return {**result, "actions": result["actions"][0]}

    def infer_batch(self, observations: list[dict]) -> dict:
        policy = self.policy
        count = len(observations)
        size = next((b for b in self.buckets if b >= count), count)
        inputs = [policy._input_transform(_copy_tree(obs)) for obs in observations]
        inputs += [inputs[-1]] * (size - count)  # padding: repeats of the last observation, results are dropped
        stacked = _stack_trees(inputs)
        if policy._is_pytorch_model:
            import torch

            stacked = _map_tree(lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device), stacked)
            rng_or_device = policy._pytorch_device
        else:
            import jax
            import jax.numpy as jnp

            stacked = _map_tree(jnp.asarray, stacked)
            policy._rng, rng_or_device = jax.random.split(policy._rng)
        start = time.monotonic()
        outputs = {
            "state": stacked["state"],
            "actions": policy._sample_actions(
                rng_or_device, self.observation_cls.from_dict(stacked), **policy._sample_kwargs
            ),
        }
        model_ms = (time.monotonic() - start) * 1000.0
        if policy._is_pytorch_model:
            outputs = _map_tree(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = _map_tree(np.asarray, outputs)
        results = [policy._output_transform(_map_tree(lambda x, i=i: x[i], outputs)) for i in range(count)]
        return {"actions": np.stack([np.asarray(r["actions"]) for r in results]), "policy_timing": {"infer_ms": model_ms}}


def _copy_tree(tree):
    return {k: _copy_tree(v) for k, v in tree.items()} if isinstance(tree, dict) else tree


def _map_tree(fn, tree):
    return {k: _map_tree(fn, v) for k, v in tree.items()} if isinstance(tree, dict) else fn(tree)


def _stack_trees(trees: list[dict]):
    first = trees[0]
    if isinstance(first, dict):
        return {k: _stack_trees([t[k] for t in trees]) for k in first}
    return np.stack([np.asarray(t) for t in trees])


def main():
    from .websocket_policy import serve_policy

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("policy", choices=["hold", "lerobot", "openpi"])
    parser.add_argument("--checkpoint", help="lerobot: local directory or Hugging Face repo id. openpi: checkpoint dir/URL.")
    parser.add_argument("--config", help="openpi: training config name, e.g. pi05_droid.")
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
    elif args.policy == "openpi":
        if not (args.checkpoint and args.config):
            parser.error("openpi needs --config and --checkpoint")
        policy = OpenPiPolicy.load(args.config, args.checkpoint)
    else:
        if not args.checkpoint:
            parser.error("lerobot needs --checkpoint")
        policy = LeRobotPolicy(args.checkpoint, dict(kv.split("=", 1) for kv in args.rename), args.device)
    serve_policy(policy, args.host, args.port)


if __name__ == "__main__":
    main()
