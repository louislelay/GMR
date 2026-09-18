"""Behavior tests for the Retargeter API."""

import numpy as np
from synthetic_motion import build_frames
from test_retarget_regression import DATA_DIR, build_retargeter


def test_retargeter_matches_golden_trajectory() -> None:
    frames = build_frames()
    retargeter = build_retargeter("unitree_g1")
    qpos = np.stack([retargeter.retarget_frame(frame) for frame in frames])

    np.testing.assert_allclose(
        qpos,
        np.load(DATA_DIR / "golden_qpos_unitree_g1.npy"),
        rtol=0.0,
        atol=1e-4,
    )


def test_retargeter_does_not_mutate_human_frame() -> None:
    frame = build_frames()[0]
    original = {
        name: (position.copy(), rotation.copy())
        for name, (position, rotation) in frame.items()
    }
    build_retargeter("unitree_g1").retarget_frame(frame)

    for name, (position, rotation) in frame.items():
        np.testing.assert_array_equal(position, original[name][0])
        np.testing.assert_array_equal(rotation, original[name][1])


def test_retargeter_generates_one_site_per_active_match() -> None:
    retargeter = build_retargeter("unitree_g1")
    profile = retargeter.profile
    plain_model = retargeter.robot.assets.load_model_spec().compile()

    active_matches = sum(
        match.position_weight != 0.0 or match.orientation_weight != 0.0
        for stage in profile.stages
        for match in stage
    )
    assert retargeter.model.nsite == plain_model.nsite + active_matches
