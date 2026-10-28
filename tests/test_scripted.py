"""Corruption definitions of the scripted expert (no simulator needed)."""

import pytest

pytest.importorskip("isaaclab.controllers")

import torch  # noqa: E402

from panda_al.scripted import (  # noqa: E402
    CORRUPTION_MODES,
    CUBE_SIZE,
    OBSTACLE_HEIGHT,
    OBSTACLE_MODE,
    OBSTACLE_RADIUS,
    sample_corruption,
)


def test_obstacle_dimensions():
    assert OBSTACLE_HEIGHT == 0.18
    assert OBSTACLE_RADIUS == CUBE_SIZE / 2


def test_obstacle_corruption_is_sampled_and_recorded():
    gen = torch.Generator().manual_seed(0)
    assert OBSTACLE_MODE in CORRUPTION_MODES
    for _ in range(20):
        c = sample_corruption([OBSTACLE_MODE], gen)
        record = c.to_dict()
        assert record["mode"] == OBSTACLE_MODE
        assert record["obstacle_radius"] == OBSTACLE_RADIUS and record["obstacle_height"] == OBSTACLE_HEIGHT
        assert 0.4 <= record["obstacle_along"] <= 0.6 and abs(record["obstacle_lateral"]) <= 0.05
        # the expert plan itself is untouched
        assert record["grasp_offset"] == [0.0, 0.0, 0.0] and record["speed_scale"] == 1.0
