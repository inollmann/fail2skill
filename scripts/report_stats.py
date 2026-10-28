"""Success rates from the ``stats.jsonl`` of collection runs (milestone M4/M5: how good is a checkpoint, where does it fail).

.. code-block:: bash

    python scripts/report_stats.py data/episodes/smolvla_iter0 [more run dirs]      # also: ./al.sh report <dirs>

One table per run directory: success rate per task with a 95 % Wilson interval, the failure tags, and the
average episode length. Works on the host (no Isaac Lab needed), ``--by_label`` splits the rows by checkpoint label.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def report(run_dir: Path, by_label: bool) -> str:
    path = run_dir / "stats.jsonl"
    if not path.exists():
        return f"{run_dir}: no stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("policy_label") if by_label else "all", row["task"])].append(row)
        groups[(row.get("policy_label") if by_label else "all", "ALL TASKS")].append(row)
    lines = [f"{run_dir}  ({len(rows)} episodes)"]
    lines.append(f"  {'policy':16s} {'task':14s} {'n':>5s} {'success':>8s} {'95% CI':>13s} {'avg steps':>10s}  failure tags")
    for (label, task), group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] == "ALL TASKS", kv[0][1])):
        n = len(group)
        wins = sum(r["outcome"] == "success" for r in group)
        low, high = wilson(wins, n)
        tags = Counter(t for r in group if r["outcome"] == "failure" for t in r.get("tags", []))
        steps = sum(r.get("steps", 0) for r in group) / n
        tag_text = ", ".join(f"{t} {c}" for t, c in tags.most_common())
        lines.append(
            f"  {str(label):16s} {task:14s} {n:5d} {100 * wins / n:7.0f}% {100 * low:5.0f}-{100 * high:3.0f}% {steps:10.0f}  {tag_text}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--by_label", action="store_true", help="Separate rows per policy label.")
    args = parser.parse_args()
    print("\n\n".join(report(Path(d), args.by_label) for d in args.run_dirs))


if __name__ == "__main__":
    main()
