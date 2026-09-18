"""Tests for immutable motion models."""

import numpy as np
import pytest

from general_motion_retargeting import HumanMotion


def test_human_motion_owns_read_only_frames() -> None:
    position = np.zeros(3)
    motion = HumanMotion(
        frames=(
            {
                "pelvis": (
                    position,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )
            },
        ),
        fps=30.0,
        height=1.8,
        source_format="test",
    )
    position[0] = 1.0

    assert motion.frames[0]["pelvis"][0][0] == 0.0
    assert motion.frame_count == 1
    assert motion.duration == pytest.approx(1.0 / 30.0)
    with pytest.raises(ValueError):
        motion.frames[0]["pelvis"][0][0] = 1.0


def test_human_motion_requires_complete_mesh_data() -> None:
    frame = {
        "pelvis": (
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    }

    with pytest.raises(ValueError, match="provided together"):
        HumanMotion(
            frames=(frame,),
            fps=30.0,
            height=1.8,
            source_format="test",
            body_vertices=np.zeros((1, 3, 3)),
        )
