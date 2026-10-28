"""Translation between the simulator and a policy family's observation / action format.

The env's action is always 8-D: 7 absolute arm joint position targets [rad] + gripper command (>= 0.5 closes).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class Observation:
    """What the simulator provides to a policy at one control step (one env)."""

    images: dict[str, np.ndarray]
    """Camera name -> (H, W, 3) uint8 RGB."""
    joint_position: np.ndarray
    """(7,) arm joint positions [rad]."""
    gripper_position: np.ndarray
    """(1,) gripper closure, 0 fully open .. 1 fully closed."""
    instruction: str


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize keeping the aspect ratio and zero-pad to (height, width), like ``openpi_client.image_tools``."""
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR))
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


def encode_jpeg(image: np.ndarray, quality: int) -> dict:
    """An RGB image as ``{"jpeg": bytes}``: about 20x smaller on the wire; the server decodes it (``decode_images``)."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return {"jpeg": buffer.getvalue()}


def decode_images(tree, pool=None):
    """Replace every ``{"jpeg": bytes}`` in a (nested) request by the decoded uint8 array. Other values are kept.

    ``pool`` (a ``concurrent.futures.Executor``) decodes several images in parallel: PIL releases the GIL.
    """
    jobs: list = []

    def walk(node):
        if isinstance(node, dict):
            if set(node) == {"jpeg"}:
                jobs.append(node["jpeg"])
                return len(jobs) - 1, True
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    def decode(data: bytes) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    skeleton = walk(tree)
    if not jobs:
        return tree
    decoded = list(pool.map(decode, jobs)) if pool is not None else [decode(j) for j in jobs]

    def fill(node):
        if isinstance(node, tuple) and len(node) == 2 and node[1] is True:
            return decoded[node[0]]
        if isinstance(node, dict):
            return {k: fill(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fill(v) for v in node]
        return node

    return fill(skeleton)


@dataclass
class PolicyAdapter:
    """Base class. ``request`` builds the observation dict sent to the server, ``to_env_action`` converts one
    row of the returned action chunk into the env's 8-D action (called at the step the row is executed)."""

    open_loop_horizon: int = 8
    """How many actions of each chunk are executed before the policy is queried again."""
    jpeg_quality: int | None = None
    """Send images as JPEG of this quality (1-100) instead of raw pixels: a remote server over Gigabit needs it
    (a request for 16 envs is about 30 MB raw). Slightly lossy, like the H.264 videos the policies are trained on."""

    def pack_image(self, image: np.ndarray):
        return image if self.jpeg_quality is None else encode_jpeg(image, self.jpeg_quality)

    def camera_names(self) -> list[str]:
        """Cameras whose images ``request`` reads."""
        raise NotImplementedError

    def request(self, obs: Observation) -> dict:
        raise NotImplementedError

    def to_env_action(self, action: np.ndarray, obs: Observation) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"adapter": type(self).__name__, **self.__dict__}


@dataclass
class NativeAdapter(PolicyAdapter):
    """This project's own format, used by :mod:`panda_al.policy.serve` (e.g. SmolVLA trained on our episodes).

    Request: ``observation/images/<camera>`` (H, W, 3) uint8, ``observation/joint_position`` (7,),
    ``observation/gripper_position`` (1,), ``observation/state`` (8,) = both concatenated, ``prompt``.
    Actions: (H, 8) in the env's action format, i.e. the same as ``data.npz["action"]`` of recorded episodes.
    """

    cameras: list[str] = field(default_factory=lambda: ["exterior_1", "exterior_2", "wrist"])

    def camera_names(self) -> list[str]:
        return list(self.cameras)

    def request(self, obs: Observation) -> dict:
        req = {f"observation/images/{name}": self.pack_image(obs.images[name]) for name in self.cameras}
        req["observation/joint_position"] = obs.joint_position.astype(np.float32)
        req["observation/gripper_position"] = obs.gripper_position.astype(np.float32)
        req["observation/state"] = np.concatenate([obs.joint_position, obs.gripper_position]).astype(np.float32)
        req["prompt"] = obs.instruction
        return req

    def to_env_action(self, action: np.ndarray, obs: Observation) -> np.ndarray:
        return np.asarray(action[:8], dtype=np.float32)


@dataclass
class DroidAdapter(PolicyAdapter):
    """openpi's DROID policies (``pi05_droid``, ``pi0_fast_droid``), following openpi's ``examples/droid``.

    Request: one exterior and the wrist image, resized with padding to 224x224, joint and gripper position.
    Actions: (H, 8) = 7 joint velocities in [-1, 1] + gripper position. As in DROID, a velocity is turned into
    a joint position target ``q + v * max_joint_delta`` relative to the joint position measured at execution.
    """

    exterior_camera: str = "exterior_1"
    wrist_camera: str = "wrist"
    image_size: int = 224
    max_joint_delta: float = 0.2
    """Joint delta [rad] per control step at velocity 1 (DROID's ``RobotIKSolver.max_joint_delta``)."""

    def camera_names(self) -> list[str]:
        return [self.exterior_camera, self.wrist_camera]

    def request(self, obs: Observation) -> dict:
        size = self.image_size
        return {
            "observation/exterior_image_1_left": self.pack_image(
                resize_with_pad(obs.images[self.exterior_camera], size, size)
            ),
            "observation/wrist_image_left": self.pack_image(resize_with_pad(obs.images[self.wrist_camera], size, size)),
            "observation/joint_position": obs.joint_position.astype(np.float64),
            "observation/gripper_position": obs.gripper_position.astype(np.float64),
            "prompt": obs.instruction,
        }

    def to_env_action(self, action: np.ndarray, obs: Observation) -> np.ndarray:
        velocity = np.clip(action[:7], -1.0, 1.0)
        target = obs.joint_position + velocity * self.max_joint_delta
        gripper = 1.0 if action[7] > 0.5 else 0.0
        return np.append(target, gripper).astype(np.float32)


ADAPTERS: dict[str, type[PolicyAdapter]] = {"native": NativeAdapter, "droid": DroidAdapter}
