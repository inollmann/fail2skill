"""Object / task registries and goal functions of the multi-task env (no simulator needed)."""

import random
import textwrap

import pytest
import torch

from panda_al import goals
from panda_al.objects import load_objects
from panda_al.taskspec import load_tasks


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def test_default_registries():
    objects = load_objects()
    assert list(objects) == ["blue_cube", "green_cube", "red_cube"]
    assert objects["red_cube"].entity == "obj_red_cube"
    assert objects["red_cube"].rest_z == pytest.approx(0.0468 / 2 + 0.0006)
    tasks = load_tasks(objects)
    assert {"stack", "beside_left", "beside_right"} <= set(tasks)
    for task in tasks.values():
        pick, place = random.choice(task.pairs(objects))
        prompt, variant = task.render_prompt(pick, place)
        assert pick.color in prompt and place.color in prompt and 0 <= variant < len(task.prompts)


def test_adding_an_object_is_one_folder(tmp_path):
    _write(
        tmp_path / "yellow_mug" / "object.yaml",
        """
        noun: mug
        color: yellow
        usd_path: mug.usd
        size: [0.08, 0.08, 0.1]
        roles: [place, distractor]
        """,
    )
    objects = load_objects(tmp_path)
    mug = objects["yellow_mug"]  # the name defaults to the folder name
    assert mug.usd_path == str((tmp_path / "yellow_mug" / "mug.usd").resolve())
    assert mug.rest_z == pytest.approx(0.05 + 0.0006)


def test_invalid_object_names_the_file(tmp_path):
    _write(tmp_path / "bad" / "object.yaml", "noun: x\ncolor: y\nusd_path: a.usd\nsize: [0.1, 0.1]\n")
    with pytest.raises(ValueError, match="bad/object.yaml"):
        load_objects(tmp_path)
    with pytest.raises(ValueError, match="no objects"):
        load_objects(tmp_path / "empty")


def test_task_filters_and_validation(tmp_path):
    objects = load_objects()
    _write(
        tmp_path / "t.yaml",
        """
        goal: {type: stack}
        pick: {color: red}
        place: {color: [blue, green]}
        prompts: ["stack {pick.color} on {place.color}"]
        """,
    )
    task = load_tasks(objects, tmp_path)["t"]  # the name defaults to the file name
    assert {(p.color, q.color) for p, q in task.pairs(objects)} == {("red", "blue"), ("red", "green")}

    _write(tmp_path / "nothing.yaml", 'goal: {type: stack}\npick: {color: purple}\nprompts: ["x"]\n')
    with pytest.raises(ValueError, match="no pair of objects"):
        load_tasks(objects, tmp_path)
    (tmp_path / "nothing.yaml").unlink()

    _write(tmp_path / "typo.yaml", 'goal: {type: stack}\nprompts: ["put {pick.colour} on {place.color}"]\n')
    with pytest.raises(AttributeError):
        load_tasks(objects, tmp_path)
    (tmp_path / "typo.yaml").unlink()

    _write(tmp_path / "goal.yaml", 'goal: {type: juggle}\nprompts: ["x"]\n')
    with pytest.raises(ValueError, match="goal type"):
        load_tasks(objects, tmp_path)


def _goal(goal: str, direction: float = 1.0, distance: float = 0.0, n: int = 1):
    ids = torch.full((n,), goals.GOAL_IDS[goal])
    return ids, torch.full((n,), direction), torch.full((n,), distance)


def test_stack_goal():
    ids, direction, distance = _goal("stack")
    size = torch.tensor([[0.0468, 0.0468, 0.0468]])
    place = torch.tensor([[0.5, 0.1, 0.0234]])
    target = goals.goal_target(ids, direction, distance, size, size, place)
    assert torch.allclose(target, torch.tensor([[0.5, 0.1, 0.0702]]))
    assert goals.goal_reached(ids, target + torch.tensor([[0.01, 0.0, 0.002]]), target).all()
    assert not goals.goal_reached(ids, target + torch.tensor([[0.0, 0.0, -0.047]]), target).all()  # beside, not on top
    assert not goals.goal_reached(ids, target + torch.tensor([[0.06, 0.0, 0.0]]), target).all()


@pytest.mark.parametrize(("direction", "sign"), [(1.0, 1.0), (-1.0, -1.0)])
def test_beside_goal(direction, sign):
    ids, direction_t, distance = _goal("beside", direction, 0.10)
    small = torch.tensor([[0.04, 0.04, 0.04]])
    big = torch.tensor([[0.08, 0.08, 0.10]])
    place = torch.tensor([[0.5, 0.0, 0.05]])  # a tall object resting on the table
    target = goals.goal_target(ids, direction_t, distance, small, big, place)
    # left = +y of the robot base frame; the small object rests on the same table plane
    assert torch.allclose(target, torch.tensor([[0.5, sign * 0.10, 0.02]]))
    assert goals.goal_reached(ids, target, target).all()
    assert not goals.goal_reached(ids, target * torch.tensor([[1.0, -1.0, 1.0]]), target).all()
