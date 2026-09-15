"""Golden-trajectory regression tests for the retargeting pipeline.

Each test retargets a deterministic synthetic human motion (see
synthetic_motion.py) and compares the resulting qpos trajectory against a
committed golden file. Any change to the IK behavior shows up as a diff
against the goldens. Intentional changes must regenerate them with

    GMR_REGEN_GOLDENS=1 uv run pytest tests/test_retarget_regression.py

then justify the diff in the pull request and attach a rendered video of the
new goldens so the motion can be certified by eye.

Set GMR_TEST_MOTION_DIR to a folder of AMASS-style SMPL-X .npz files to also
run the (local-only, license-gated) real-motion smoke test.
"""

import os
import pathlib

import numpy as np
import pytest
from synthetic_motion import build_frames

from general_motion_retargeting import GeneralMotionRetargeting

DATA_DIR = pathlib.Path(__file__).parent / "data"

ROBOTS = ["unitree_g1", "booster_t1"]

# Cross-platform tolerance: QP solves accumulate small floating-point
# differences across BLAS implementations. Real regressions are orders of
# magnitude larger.
ATOL = 1e-4

# Convergence bound (meters) on the position error of high-weight IK tasks.
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
    """The IK must actually reach the targets it was asked to track.

    Unlike the golden comparison, this is an absolute correctness check: for
    every frame, the position error of each tier-one frame task (position
    weight >= 50: pelvis/waist and feet) must converge below TRACKING_ATOL.
    Lower-weight posture tasks (elbows, knees, ...) are excluded: they guide
    the pose but are unreachable by design when human and robot limb
    proportions differ. This test fails when the solver loop, target scaling,
    or task offsets break, independent of any golden data.
    """
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


@pytest.mark.skipif(
    "GMR_TEST_MOTION_DIR" not in os.environ,
    reason="set GMR_TEST_MOTION_DIR to a folder of SMPL-X .npz motions",
)
def test_real_motion_smoke():
    """Local-only smoke test on real (non-redistributable) SMPL-X motions.

    AMASS and LAFAN1 licensing does not allow committing motion data to the
    repo, so this test only runs on machines that point GMR_TEST_MOTION_DIR
    at a local folder of AMASS-style SMPL-X .npz files.
    """
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    motion_dir = pathlib.Path(os.environ["GMR_TEST_MOTION_DIR"])
    body_models = pathlib.Path(__file__).parents[1] / "assets" / "body_models"
    motions = sorted(motion_dir.rglob("*.npz"))[:2]
    assert motions, f"no .npz motions found under {motion_dir}"
    for motion_file in motions:
        smplx_data, body_model, smplx_output, human_height = load_smplx_file(
            motion_file, str(body_models)
        )
        frames, _ = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
        retargeter = GeneralMotionRetargeting(
            "smplx", "unitree_g1", actual_human_height=human_height, verbose=False
        )
        qpos = np.stack([retargeter.retarget(frame) for frame in frames[:60]])
        assert np.isfinite(qpos).all()
