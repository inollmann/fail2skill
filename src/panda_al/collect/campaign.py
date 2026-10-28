"""Campaign files: what an unattended collection run does (see ``configs/campaigns/``).

Pure Python, validated on load so that a typo fails at start-up and not after hours.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..policy.adapters import ADAPTERS
from ..scripted import CORRUPTION_MODES, OBSTACLE_MODE

OUTCOMES = ("success", "failure")


@dataclass
class PolicySettings:
    host: str = field(default_factory=lambda: os.environ.get("AL_POLICY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("AL_POLICY_PORT", "8000")))
    adapter: str = "native"
    open_loop_horizon: int | None = None
    jpeg_quality: int | None = None
    """Send images as JPEG (e.g. 90) instead of raw pixels: needed for a remote server over Gigabit."""
    label: str = "policy"
    """Name of the checkpoint / policy version; stored in every episode and in the stats."""
    batch: str | bool = "auto"
    """``auto``: one batched request per step if the server supports it (otherwise one connection per env),
    ``true``: require it, ``false``: one connection per env."""
    reconnect_attempts: int = 30
    """Tries (2 s apart) to get a lost policy server back before the run stops with exit code 75."""


@dataclass
class ExpertSettings:
    corrupt_ratio: float = 0.5
    modes: list[str] = field(default_factory=lambda: sorted(set(CORRUPTION_MODES) - {OBSTACLE_MODE}))


@dataclass
class SaveSettings:
    outcomes: list[str] = field(default_factory=lambda: ["failure"])
    keep_success_fraction: float = 0.0
    """Fraction of successful episodes saved anyway (if ``success`` is not in ``outcomes``)."""
    drop_tags: list[str] = field(default_factory=lambda: ["stalled_at_start"])
    quota_per_task: int | None = None
    quota_per_tag: int | None = None


@dataclass
class Limits:
    max_episodes: int | None = None
    max_hours: float | None = None
    max_disk_gb: float | None = None


@dataclass
class Campaign:
    name: str
    driver: str = "policy"
    env: str = "PandaAL-MultiTask-v0"
    num_envs: int = 8
    seed: int = 0
    tasks: dict[str, float] | None = None
    """Task name -> weight (default: all tasks with the weights of their YAML files)."""
    cameras: list[str] = field(default_factory=lambda: ["exterior_1", "exterior_2", "wrist"])
    depth: bool = False
    obstacle_probability: float = 0.0
    policy: PolicySettings = field(default_factory=PolicySettings)
    expert: ExpertSettings = field(default_factory=ExpertSettings)
    save: SaveSettings = field(default_factory=SaveSettings)
    limits: Limits = field(default_factory=Limits)

    @property
    def output_dir(self) -> str:
        return os.path.join(os.environ.get("AL_DATA_DIR", "data"), "episodes", self.name)


def _section(cls, data: dict | None, where: str):
    data = dict(data or {})
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(cls.__dataclass_fields__)})")
    return cls(**data)


def parse_campaign(data: dict) -> Campaign:
    data = dict(data)
    try:
        perturbations = data.pop("perturbations", None) or {}
        unknown = set(perturbations) - {"obstacle"}
        if unknown:
            raise ValueError(f"perturbations: unknown {sorted(unknown)} (implemented: obstacle)")
        save = data.pop("save", None) or {}
        quota = save.pop("quota", None) or {}
        save = {
            **save,
            **({"quota_per_task": quota["per_task"]} if "per_task" in quota else {}),
            **({"quota_per_tag": quota["per_tag"]} if "per_tag" in quota else {}),
        }
        sections = {
            "policy": _section(PolicySettings, data.pop("policy", None), "policy"),
            "expert": _section(ExpertSettings, data.pop("expert", None), "expert"),
            "save": _section(SaveSettings, save, "save"),
            "limits": _section(Limits, data.pop("limits", None), "limits"),
        }
        unknown = set(data) - set(Campaign.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)}")
        campaign = Campaign(
            **data,
            **sections,
            obstacle_probability=float((perturbations.get("obstacle") or {}).get("probability", 0.0)),
        )
    except TypeError as err:
        raise ValueError(f"invalid campaign: {err}") from err

    if campaign.driver not in ("policy", "expert"):
        raise ValueError("driver must be 'policy' or 'expert'")
    if isinstance(campaign.policy.batch, bool):  # YAML turns an unquoted true / false into a boolean
        campaign.policy.batch = "true" if campaign.policy.batch else "false"
    if campaign.policy.batch not in ("auto", "true", "false"):
        raise ValueError("policy.batch must be auto, true or false")
    if campaign.policy.adapter not in ADAPTERS:
        raise ValueError(f"policy.adapter must be one of {sorted(ADAPTERS)}")
    bad = set(campaign.save.outcomes) - set(OUTCOMES)
    if bad or not campaign.save.outcomes:
        raise ValueError(f"save.outcomes must be a non-empty subset of {OUTCOMES}")
    if not 0.0 <= campaign.save.keep_success_fraction <= 1.0 or not 0.0 <= campaign.obstacle_probability <= 1.0:
        raise ValueError("probabilities and fractions must be in [0, 1]")
    unknown_modes = set(campaign.expert.modes) - set(CORRUPTION_MODES)
    if unknown_modes:
        raise ValueError(f"expert.modes: unknown {sorted(unknown_modes)}")
    if campaign.num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    return campaign


def load_campaign(path: str | Path) -> Campaign:
    path = Path(path)
    try:
        return parse_campaign(yaml.safe_load(path.read_text()) or {})
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err
