"""Locations of the repo's data directories (also valid in the containers, where ``src/`` sits next to them)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
OBJECTS_DIR = ASSETS_DIR / "objects"
TASKS_DIR = REPO_ROOT / "configs" / "tasks"
