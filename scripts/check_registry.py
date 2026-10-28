"""Validate the object and task registries and print what they define (no simulator needed).

.. code-block:: bash

    ./al.sh registry                      # same as: ./al.sh run python scripts/check_registry.py

Run it after adding a folder to ``assets/objects/`` or a file to ``configs/tasks/``. It loads every definition, checks
that local USD files exist and that each task has valid object pairs and well-formed prompt templates.
"""

import argparse
import random
from pathlib import Path

from panda_al.objects import OBJECTS_DIR, load_objects
from panda_al.taskspec import TASKS_DIR, load_tasks

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--objects", default=str(OBJECTS_DIR))
parser.add_argument("--tasks", default=str(TASKS_DIR))
args = parser.parse_args()

objects = load_objects(args.objects)
print(f"{len(objects)} objects in {args.objects}")
problems = 0
for obj in objects.values():
    local = "{" not in obj.usd_path and "://" not in obj.usd_path
    missing = local and not Path(obj.usd_path).exists()
    problems += missing
    size = "x".join(f"{v * 100:.1f}" for v in obj.size)
    print(f"  {obj.name:16s} {obj.color:8s} {obj.noun:8s} {size} cm  roles={','.join(obj.roles)}"
          f"{'   MISSING USD: ' + obj.usd_path if missing else ''}")

tasks = load_tasks(objects, args.tasks)
print(f"{len(tasks)} tasks in {args.tasks}")
for task in tasks.values():
    pairs = task.pairs(objects)
    pick, place = random.choice(pairs)
    print(f"  {task.name:16s} goal={task.goal['type']:7s} {len(pairs)} object pairs, weight {task.weight}")
    print(f"      e.g. \"{task.render_prompt(pick, place)[0]}\"")
raise SystemExit(1 if problems else 0)
