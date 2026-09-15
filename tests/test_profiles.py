"""Tests for typed retargeting profiles."""

import json
from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting import build_catalog, load_profile


@pytest.mark.parametrize(
    ("source_format", "robot", "path"),
    [
        (source_format, robot, reference.path)
        for (source_format, robot), reference in build_catalog(
            discover=False
        ).profiles.items()
    ],
)
def test_loads_every_builtin_profile(
    source_format: str, robot: str, path: Path
) -> None:
    profile = load_profile(path, source_format=source_format, robot=robot)
    assert profile.stages


def test_loads_v2_named_matches(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "identifier": "test_profile",
                "robot_root_name": "base",
                "human_root_name": "pelvis",
                "ground_height": 0.0,
                "human_height_assumption": 1.8,
                "human_scale_table": {"pelvis": 1.0},
                "stages": [
                    {
                        "root": {
                            "robot": {"body": "base"},
                            "human": {"body": "pelvis"},
                            "position_weight": 100.0,
                            "orientation_weight": 10.0,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    profile = load_profile(path, source_format="smplx", robot="test")

    assert profile.identifier == "test_profile"
    assert profile.stages[0][0].robot.body == "base"
    np.testing.assert_array_equal(
        profile.stages[0][0].robot.rotation, [1.0, 0.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError):
        profile.stages[0][0].human.position[0] = 1.0


def test_rejects_non_unit_profile_quaternion(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "robot_root_name": "base",
                "human_root_name": "pelvis",
                "ground_height": 0.0,
                "human_height_assumption": 1.8,
                "human_scale_table": {"pelvis": 1.0},
                "stages": [
                    {
                        "root": {
                            "robot": {"body": "base"},
                            "human": {
                                "body": "pelvis",
                                "rotation": [0.0, 0.0, 0.0, 0.0],
                            },
                            "position_weight": 1.0,
                            "orientation_weight": 1.0,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unit wxyz"):
        load_profile(path, source_format="smplx", robot="test")
