"""Tests for canonical RobotMotion validation and persistence."""

from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting import (
    RobotMotion,
    load_robot_motion,
    save_robot_motion,
)


def _motion() -> RobotMotion:
    return RobotMotion(
        robot="test_robot",
        source_format="smplx",
        profile="test_profile",
        fps=20.0,
        root_positions=np.arange(15, dtype=np.float64).reshape(5, 3),
        root_quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
        joint_positions=np.arange(10, dtype=np.float64).reshape(5, 2),
        joint_names=("left", "right"),
        source_identifier="sha256:abc",
        source_fps=30.0,
    )


def test_round_trip_uses_pickle_free_npz(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"

    save_robot_motion(path, _motion())
    loaded = load_robot_motion(path)

    assert loaded.robot == "test_robot"
    assert loaded.joint_names == ("left", "right")
    assert loaded.source_identifier == "sha256:abc"
    np.testing.assert_array_equal(loaded.qpos, _motion().qpos)
    with np.load(path, allow_pickle=False) as data:
        assert data["joint_names"].dtype.kind == "U"


def test_robot_motion_owns_read_only_arrays() -> None:
    positions = np.zeros((1, 3))
    motion = RobotMotion(
        robot="test",
        source_format="test",
        profile="test",
        fps=30.0,
        root_positions=positions,
        root_quaternions=np.array([[1.0, 0.0, 0.0, 0.0]]),
        joint_positions=np.empty((1, 0)),
        joint_names=(),
    )
    positions[0, 0] = 1.0

    assert motion.root_positions[0, 0] == 0.0
    with pytest.raises(ValueError):
        motion.root_positions[0, 0] = 1.0


def test_trim_uses_seconds_and_preserves_source_motion() -> None:
    motion = _motion()

    trimmed = motion.trim(0.05, 0.2)

    assert trimmed.frame_count == 3
    assert trimmed.start_time == pytest.approx(0.05)
    assert motion.frame_count == 5


def test_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    save_robot_motion(path, _motion())
    with np.load(path, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
    fields["schema_version"] = np.asarray(99)
    np.savez(path, **fields)

    with pytest.raises(ValueError, match="unsupported"):
        load_robot_motion(path)
