"""Golden-trajectory regression tests for the retargeting pipeline.

Regenerate the expected trajectories with:

    GMR_REGEN_GOLDENS=1 uv run pytest tests/test_retarget_regression.py
"""

import os
import pathlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_motion import Frame, build_frames

from general_motion_retargeting import (
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    Retargeter,
    RobotSpec,
    SolverSettings,
    TrackingCamera,
    load_profile,
)
from general_motion_retargeting.assets import PathRobotAssets

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / "data"

ROBOTS: tuple[str, ...] = ("unitree_g1", "booster_t1")
# Allow small differences between BLAS implementations.
ATOL: float = 1e-4


def load_synthetic_motion() -> tuple[Frame, ...]:
    """Load the deterministic human motion used by regression tests."""
    return build_frames()


def build_retargeter(robot: str, settings: SolverSettings | None = None) -> Retargeter:
    """Build a retargeter from the current explicit contracts."""
    profile = load_profile(
        pathlib.Path(IK_CONFIG_DICT["smplx"][robot]),
        source_format="smplx",
        robot=robot,
    )
    root = str(ROBOT_BASE_DICT[robot])
    specification = RobotSpec(
        identifier=robot,
        assets=PathRobotAssets(pathlib.Path(ROBOT_XML_DICT[robot])),
        root_body=root,
        camera=TrackingCamera(body=root, distance=3.0),
    )
    return Retargeter(specification, profile, settings)


def retarget_motion(robot: str, frames: Sequence[Frame]) -> NDArray[np.float64]:
    """Retarget a sequence of human frames to one robot."""
    retargeter = build_retargeter(robot)
    return np.stack([retargeter.retarget_frame(frame) for frame in frames])


@pytest.fixture(scope="module")
def synthetic_frames() -> tuple[Frame, ...]:
    """Provide deterministic human motion frames."""
    return load_synthetic_motion()


@pytest.mark.parametrize("robot", ROBOTS)
def test_qpos_matches_golden(robot: str, synthetic_frames: Sequence[Frame]) -> None:
    qpos = retarget_motion(robot, synthetic_frames)
    golden_path = DATA_DIR / f"golden_qpos_{robot}.npy"
    if os.environ.get("GMR_REGEN_GOLDENS") == "1":
        np.save(golden_path, qpos)
        pytest.skip(f"regenerated {golden_path.name}; verify the diff and commit")
    golden = np.load(golden_path)
    assert qpos.shape == golden.shape
    np.testing.assert_allclose(qpos, golden, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("robot", ROBOTS)
def test_qpos_is_sane(robot: str, synthetic_frames: Sequence[Frame]) -> None:
    qpos = retarget_motion(robot, synthetic_frames)
    assert np.isfinite(qpos).all()
    # The synthetic subject stands upright; the floating-base height must
    # stay in a plausible standing range for a human-scale robot.
    assert (qpos[:, 2] > 0.3).all()
    assert (qpos[:, 2] < 1.5).all()
    # The root quaternion must stay normalized.
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
