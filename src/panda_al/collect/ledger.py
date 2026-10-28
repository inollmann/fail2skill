"""Bookkeeping of a collection campaign: what was collected, what is kept, when to stop.

Two append-only JSON-lines files in the run directory make a campaign resumable and auditable:

``stats.jsonl``     one line per finished episode, kept or not (success rates need the successes too)
``manifest.jsonl``  one line per saved episode (the dataset index); ``path`` is relative to the run directory

Episodes that are kept are written to disk in the background. Their lines go into the files only once the episode is
complete on disk (the callable returned by :meth:`Ledger.record`), so after a crash the files only list what exists;
quota counts include the episodes still being written.

No simulator needed.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path


class Ledger:
    def __init__(
        self,
        run_dir: str | Path,
        tasks: list[str],
        quota_per_task: int | None = None,
        quota_per_tag: int | None = None,
        drop_tags: list[str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stats_path = self.run_dir / "stats.jsonl"
        self.manifest_path = self.run_dir / "manifest.jsonl"
        self._lock = threading.Lock()
        self.tasks = tasks
        self.quota_per_task = quota_per_task
        self.quota_per_tag = quota_per_tag
        self.drop_tags = set(drop_tags or [])
        self.finished = Counter()
        """outcome -> count of finished episodes."""
        self.saved_per_task = Counter()
        self.saved_per_tag = Counter()
        self.outcomes_per_task: dict[str, Counter] = {}
        for line in self._read(self.stats_path):
            self.finished[line["outcome"]] += 1
            self.outcomes_per_task.setdefault(line["task"], Counter())[line["outcome"]] += 1
        for line in self._read(self.manifest_path):
            self.saved_per_task[line["task"]] += 1
            self.saved_per_tag.update(line.get("tags", []))

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @property
    def total_finished(self) -> int:
        return sum(self.finished.values())

    @property
    def total_saved(self) -> int:
        return sum(self.saved_per_task.values())

    def decide(self, task: str, tags: list[str]) -> tuple[bool, str]:
        """Whether a *failed* episode with these tags is kept, and why (not)."""
        if tags and set(tags) <= self.drop_tags:
            return False, "dropped_tags"
        if self.quota_per_task is not None and self.saved_per_task[task] >= self.quota_per_task:
            return False, "task_quota_full"
        if self.quota_per_tag is not None and tags and all(self.saved_per_tag[t] >= self.quota_per_tag for t in tags):
            return False, "tag_quota_full"
        return True, "kept"

    def record(self, entry: dict, kept: bool, reason: str) -> Callable[[], None]:
        """Count a finished episode. ``entry`` needs ``task``, ``outcome`` and optionally ``tags`` and anything else.

        Episodes that are not kept are logged at once. For kept ones the returned callable writes the log lines; call it
        when the episode is on disk. (It is safe to call from another thread.)
        """
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry, "kept": kept, "reason": reason}
        self.finished[entry["outcome"]] += 1
        self.outcomes_per_task.setdefault(entry["task"], Counter())[entry["outcome"]] += 1
        if not kept:
            self._append(self.stats_path, entry)
            return lambda: None
        self.saved_per_task[entry["task"]] += 1
        self.saved_per_tag.update(entry.get("tags", []))

        def persist():
            self._append(self.stats_path, entry)
            self._append(self.manifest_path, entry)

        return persist

    def _append(self, path: Path, entry: dict):
        with self._lock, open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def quotas_met(self) -> bool:
        """Every selected task has reached its per-task quota (never true without one)."""
        return self.quota_per_task is not None and all(self.saved_per_task[t] >= self.quota_per_task for t in self.tasks)

    def success_rates(self) -> dict[str, float]:
        """Per task: successes / finished episodes of this campaign (the weakness signal)."""
        return {
            task: counts["success"] / sum(counts.values()) for task, counts in self.outcomes_per_task.items() if counts
        }

    def summary(self) -> str:
        rates = ", ".join(f"{t} {100 * r:.0f}%" for t, r in sorted(self.success_rates().items()))
        saved = ", ".join(f"{t} {self.saved_per_task[t]}" for t in self.tasks)
        return (
            f"{self.total_finished} finished ({self.finished['success']} success, {self.finished['failure']} failure), "
            f"{self.total_saved} saved [{saved}]; success rate: {rates or 'n/a'}"
        )
