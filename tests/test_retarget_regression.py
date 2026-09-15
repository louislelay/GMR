"""Golden-trajectory regression tests for the retargeting pipeline.

Regenerate the expected trajectories with:

    GMR_REGEN_GOLDENS=1 uv run pytest tests/test_retarget_regression.py
"""

import os
import pathlib

import numpy as np
import pytest
from synthetic_motion import build_frames

from general_motion_retargeting import GeneralMotionRetargeting

DATA_DIR = pathlib.Path(__file__).parent / "data"

ROBOTS = ["unitree_g1", "booster_t1"]

# Allow small differences between BLAS implementations.
ATOL = 1e-4

# Maximum position error in meters for high-weight IK tasks.
TRACKING_ATOL = 0.05


def load_synthetic_motion():
    return build_frames()


def retarget_motion(robot, frames):
    retargeter = GeneralMotionRetargeting("smplx", robot, verbose=False)
    return np.stack([retargeter.retarget(dict(frame)) for frame in frames])


@pytest.fixture(scope="module")
def synthetic_frames():
    return load_synthetic_motion()


@pytest.mark.parametrize("robot", ROBOTS)
def test_qpos_matches_golden(robot, synthetic_frames):
    qpos = retarget_motion(robot, synthetic_frames)
    golden_path = DATA_DIR / f"golden_qpos_{robot}.npy"
    if os.environ.get("GMR_REGEN_GOLDENS"):
        np.save(golden_path, qpos)
        pytest.skip(f"regenerated {golden_path.name}; verify the diff and commit")
    golden = np.load(golden_path)
    assert qpos.shape == golden.shape
    np.testing.assert_allclose(qpos, golden, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("robot", ROBOTS)
def test_qpos_is_sane(robot, synthetic_frames):
    qpos = retarget_motion(robot, synthetic_frames)
    assert np.isfinite(qpos).all()
    # The synthetic subject stands upright; the floating-base height must
    # stay in a plausible standing range for a human-scale robot.
    assert (qpos[:, 2] > 0.3).all()
    assert (qpos[:, 2] < 1.5).all()
    # The root quaternion must stay normalized.
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


@pytest.mark.parametrize("robot", ROBOTS)
def test_high_weight_tasks_converge(robot, synthetic_frames):
    """Check that high-weight position tasks converge."""
    retargeter = GeneralMotionRetargeting("smplx", robot, verbose=False)
    tasks = [
        task
        for task in retargeter.tasks1 + retargeter.tasks2
        if np.max(task.cost[:3]) >= 50.0
    ]
    assert tasks, "expected at least one high-weight position task"
    worst = 0.0
    for frame in synthetic_frames:
        retargeter.retarget(dict(frame))
        for task in tasks:
            error = task.compute_error(retargeter.configuration)
            worst = max(worst, float(np.linalg.norm(error[:3])))
    assert worst < TRACKING_ATOL, (
        f"high-weight IK tasks did not converge: worst position error "
        f"{worst:.4f} m exceeds {TRACKING_ATOL} m"
    )
