"""Object registry: every folder in ``assets/objects/`` with an ``object.yaml`` is a manipulable object.

Pure Python (no Isaac Lab), so it can be used by tooling outside the simulator. See ``docs/automated_collection.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import OBJECTS_DIR

ROLES = ("pick", "place", "distractor")
SPAWN_MARGIN = 0.0006
"""Objects spawn this far above the table so they do not start in contact [m]."""


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    noun: str
    """Word used in prompts, e.g. ``cube``."""
    color: str
    usd_path: str
    """Absolute path, ``{ISAAC_NUCLEUS_DIR}/...``, or relative to the object's folder (resolved on load)."""
    size: tuple[float, float, float]
    """Bounding box [m] of the object at ``scale``."""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roles: tuple[str, ...] = ROLES
    mass: float | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    spawn_height: float | None = None

    @property
    def rest_z(self) -> float:
        """Height of the object's origin above the table when it lies on it."""
        return self.spawn_height if self.spawn_height is not None else self.size[2] / 2 + SPAWN_MARGIN

    @property
    def entity(self) -> str:
        """Name of the scene entity."""
        return f"obj_{self.name}"

    def resolved_usd_path(self) -> str:
        """USD path with the Nucleus placeholder expanded (needs Isaac Lab)."""
        if "{ISAAC_NUCLEUS_DIR}" in self.usd_path:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            return self.usd_path.replace("{ISAAC_NUCLEUS_DIR}", ISAAC_NUCLEUS_DIR)
        return self.usd_path


def _spec_from_yaml(path: Path) -> ObjectSpec:
    data = yaml.safe_load(path.read_text())
    try:
        usd = str(data["usd_path"])
        if "{" not in usd and "://" not in usd and not Path(usd).is_absolute():
            usd = str((path.parent / usd).resolve())
        roles = tuple(data.get("roles", ROLES))
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)} (allowed: {ROLES})")
        size = tuple(float(v) for v in data["size"])
        if len(size) != 3 or min(size) <= 0:
            raise ValueError("size must be three positive numbers [m]")
        return ObjectSpec(
            name=str(data.get("name", path.parent.name)),
            noun=str(data["noun"]),
            color=str(data["color"]),
            usd_path=usd,
            size=size,
            scale=tuple(float(v) for v in data.get("scale", (1.0, 1.0, 1.0))),
            roles=roles,
            mass=data.get("mass"),
            static_friction=data.get("static_friction"),
            dynamic_friction=data.get("dynamic_friction"),
            spawn_height=data.get("spawn_height"),
        )
    except (KeyError, ValueError, TypeError) as err:
        raise ValueError(f"{path}: invalid object definition: {err!r}") from err


def load_objects(root: str | Path = OBJECTS_DIR) -> dict[str, ObjectSpec]:
    """All objects below ``root``, in alphabetical order of their folder (this fixes the order in the scene)."""
    objects: dict[str, ObjectSpec] = {}
    for manifest in sorted(Path(root).glob("*/object.yaml")):
        spec = _spec_from_yaml(manifest)
        if spec.name in objects:
            raise ValueError(f"{manifest}: duplicate object name {spec.name!r}")
        objects[spec.name] = spec
    if not objects:
        raise ValueError(f"no objects found in {root} (expected <name>/object.yaml)")
    return objects
