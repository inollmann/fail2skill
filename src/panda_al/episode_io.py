"""On-disk episode format shared by the simulator, the replay tool and the Cosmos side.

Deliberately free of Isaac Lab / torch imports so it can be used in any Python environment
(e.g. a Cosmos post-training container) with only ``numpy`` and ``av`` installed.

Layout of one run directory::

    <run_dir>/
        success/<episode_id>/
        failure/<episode_id>/
        in_progress/<episode_id>/          # episodes still being written (or from a crashed run)

Layout of one episode directory::

    meta.json            # see EpisodeMeta below
    data.npz             # per-step arrays, all with leading dimension T (see STEP_KEYS)
    initial_state.npz    # Isaac Lab scene state (flattened ``scene.get_state(is_relative=True)``)
    videos/<camera>.mp4  # one H.264 video per camera, exactly T frames
    videos/<camera>_depth.mkv  # optional, lossless 16-bit grayscale depth in mm (0 = invalid), T frames

Row ``t`` of ``data.npz`` pairs the observation *before* executing ``action[t]`` with that action
(the DROID convention). The last row (``is_last == True``) holds the terminal observation; its
action repeats the previous one and should not be used for training.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUCCESS_DIR = "success"
FAILURE_DIR = "failure"
IN_PROGRESS_DIR = "in_progress"

STEP_KEYS: dict[str, str] = {
    "action": "(T, 8) float32 - absolute arm joint position targets [rad] (7) + gripper command (0 open, 1 closed)",
    "joint_position": "(T, 7) float32 - measured arm joint positions [rad]",
    "joint_velocity": "(T, 7) float32 - measured arm joint velocities [rad/s]",
    "gripper_position": "(T, 1) float32 - gripper closure, 0 fully open .. 1 fully closed",
    "cartesian_position": "(T, 6) float32 - TCP pose in base frame: xyz [m] + roll/pitch/yaw [rad] (XYZ extrinsic)",
    "ee_quat": "(T, 4) float32 - TCP orientation in robot base frame, quaternion (x, y, z, w)",
    "object_pose": "(T, K, 7) float32 - object poses relative to env origin: xyz + quat (x, y, z, w)",
    "timestamp": "(T,) float64 - simulation time since episode start [s]",
    "is_first": "(T,) bool",
    "is_last": "(T,) bool",
}


@dataclass
class EpisodeMeta:
    """Metadata written to ``meta.json``."""

    episode_id: str
    task: str
    instruction: str
    source: str
    """How the episode was produced: ``teleop``, ``scripted``, ``replay``, ``synthetic`` ..."""
    fps: float
    cameras: list[str]
    object_names: list[str]
    arm_joint_names: list[str]
    depth_cameras: list[str] = field(default_factory=list)
    """Cameras that also have a ``<camera>_depth.mkv`` video."""
    success: bool | None = None
    outcome: str | None = None
    """``success``, ``failure`` or ``discard``."""
    termination_terms: list[str] = field(default_factory=list)
    """Isaac Lab termination terms that fired at the end of the episode (e.g. ``time_out``)."""
    num_steps: int = 0
    parent_episode: str | None = None
    """For replays / synthetic episodes: the episode this one was derived from."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form producer info, e.g. the injected corruption and its parameters."""

    @classmethod
    def from_dict(cls, data: dict) -> EpisodeMeta:
        known = {f for f in cls.__dataclass_fields__}
        meta = cls(**{k: v for k, v in data.items() if k in known})
        meta.extra.update({k: v for k, v in data.items() if k not in known})
        return meta


def _pick_codec(preferred: tuple[str, ...]) -> str:
    import av

    for name in preferred:
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError(f"None of the video codecs {preferred} is available in this PyAV build.")


class VideoStream:
    """Streams RGB frames into an mp4 file so long episodes never sit in memory."""

    CODECS = ("libx264", "libopenh264", "h264", "mpeg4")

    def __init__(self, path: Path, fps: float, width: int, height: int, crf: int = 20):
        from fractions import Fraction

        import av

        self._container = av.open(str(path), mode="w")
        codec = _pick_codec(self.CODECS)
        self._stream = self._container.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            self._stream.options = {"crf": str(crf), "preset": "veryfast"}
        self._av = av

    def write(self, rgb: np.ndarray):
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class DepthStream:
    """Streams 16-bit depth frames (mm) into a lossless FFV1 matroska file."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        from fractions import Fraction

        import av

        self._container = av.open(str(path), mode="w", format="matroska")
        self._stream = self._container.add_stream("ffv1", rate=Fraction(fps).limit_denominator(1000))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "gray16le"
        self._av = av

    def write(self, depth_mm: np.ndarray):
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(depth_mm, dtype=np.uint16), format="gray16le")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class EpisodeWriter:
    """Writes one episode into ``<run_dir>/in_progress/<id>`` and moves it on :meth:`finalize`.

    Video encoding runs on a background thread so the simulation loop only pays for the
    GPU -> CPU copy of the frames.
    """

    def __init__(self, run_dir: str | os.PathLike, meta: EpisodeMeta, max_queue: int = 64):
        self.run_dir = Path(run_dir)
        self.meta = meta
        self.dir = self.run_dir / IN_PROGRESS_DIR / meta.episode_id
        (self.dir / "videos").mkdir(parents=True, exist_ok=True)
        self._steps: dict[str, list[np.ndarray]] = {}
        self._rows = 0
        self._videos: dict[str, VideoStream | DepthStream] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._thread.start()

    @property
    def num_rows(self) -> int:
        return self._rows

    def set_initial_state(self, state: dict[str, np.ndarray]):
        np.savez_compressed(self.dir / "initial_state.npz", **state)

    def append(
        self,
        frames: dict[str, np.ndarray],
        step: dict[str, np.ndarray],
        depth: dict[str, np.ndarray] | None = None,
    ):
        """Append one row: an RGB frame (H, W, 3) per camera, optional uint16 depth (H, W) in mm, and the step arrays."""
        if self._error is not None:
            raise RuntimeError(f"Video encoding failed for episode {self.meta.episode_id}") from self._error
        for key, value in step.items():
            self._steps.setdefault(key, []).append(np.asarray(value))
        self._rows += 1
        self._queue.put((frames, depth or {}))

    def _encode_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frames, depth = item
            try:
                for name, mm in depth.items():
                    if f"{name}_depth" not in self._videos:
                        h, w = mm.shape[:2]
                        self._videos[f"{name}_depth"] = DepthStream(
                            self.dir / "videos" / f"{name}_depth.mkv", self.meta.fps, w, h
                        )
                    self._videos[f"{name}_depth"].write(mm)
                for name, rgb in frames.items():
                    if name not in self._videos:
                        h, w = rgb.shape[:2]
                        self._videos[name] = VideoStream(self.dir / "videos" / f"{name}.mp4", self.meta.fps, w, h)
                    self._videos[name].write(rgb)
            except BaseException as err:  # surfaced on the next append / finalize
                self._error = err

    def _close_videos(self):
        self._queue.put(None)
        self._thread.join()
        for video in self._videos.values():
            video.close()
        if self._error is not None:
            raise RuntimeError(f"Video encoding failed for episode {self.meta.episode_id}") from self._error

    def finalize(self, outcome: str, termination_terms: list[str] | None = None) -> Path | None:
        """Flush everything and move the episode to ``success/`` or ``failure/`` (or delete it on ``discard``)."""
        self._close_videos()
        if outcome == "discard" or self.num_rows == 0:
            shutil.rmtree(self.dir, ignore_errors=True)
            return None
        arrays = {key: np.stack(values) for key, values in self._steps.items()}
        np.savez_compressed(self.dir / "data.npz", **arrays)
        self.meta.success = outcome == "success"
        self.meta.outcome = outcome
        self.meta.termination_terms = list(termination_terms or [])
        self.meta.num_steps = self.num_rows
        (self.dir / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        target = self.run_dir / (SUCCESS_DIR if outcome == "success" else FAILURE_DIR) / self.meta.episode_id
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.dir), target)
        return target


@dataclass
class Episode:
    """An episode loaded from disk (videos are referenced by path, not decoded)."""

    path: Path
    meta: EpisodeMeta
    data: dict[str, np.ndarray]
    initial_state: dict[str, np.ndarray]

    @property
    def videos(self) -> dict[str, Path]:
        return {name: self.path / "videos" / f"{name}.mp4" for name in self.meta.cameras}

    def read_depth(self, camera: str) -> np.ndarray:
        """Decode a depth video into a (T, H, W) uint16 array in mm (0 = invalid)."""
        import av

        with av.open(str(self.path / "videos" / f"{camera}_depth.mkv")) as container:
            return np.stack([f.to_ndarray(format="gray16le") for f in container.decode(video=0)])

    def read_video(self, camera: str) -> np.ndarray:
        """Decode a camera video into a (T, H, W, 3) uint8 array."""
        import av

        with av.open(str(self.videos[camera])) as container:
            return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def load_episode(path: str | os.PathLike) -> Episode:
    path = Path(path)
    meta = EpisodeMeta.from_dict(json.loads((path / "meta.json").read_text()))
    with np.load(path / "data.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    state_file = path / "initial_state.npz"
    initial_state = {}
    if state_file.exists():
        with np.load(state_file) as npz:
            initial_state = {k: npz[k] for k in npz.files}
    return Episode(path=path, meta=meta, data=data, initial_state=initial_state)


def iter_episodes(run_dir: str | os.PathLike, outcome: str | None = None):
    """Yield episode directories below ``run_dir`` (optionally only ``success`` or ``failure``)."""
    run_dir = Path(run_dir)
    subdirs = [SUCCESS_DIR, FAILURE_DIR] if outcome is None else [outcome]
    for sub in subdirs:
        base = run_dir / sub
        if base.is_dir():
            yield from sorted(p for p in base.iterdir() if (p / "meta.json").exists())


def flatten_state(state: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested ``scene.get_state()`` dict into ``{"articulation/robot/joint_position": array}``."""
    flat = {}
    for key, value in state.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_state(value, name))
        else:
            flat[name] = np.asarray(value)
    return flat


def unflatten_state(flat: dict[str, np.ndarray]) -> dict:
    """Inverse of :func:`flatten_state`."""
    state: dict = {}
    for name, value in flat.items():
        node = state
        *parents, leaf = name.split("/")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return state
