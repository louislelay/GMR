"""Tests for versioned NPZ motion save/load and legacy pickle fallback."""

import pickle

import numpy as np
import pytest
from test_retarget_regression import DATA_DIR

from general_motion_retargeting import load_robot_motion, save_robot_motion
from general_motion_retargeting.motion_io import SCHEMA_VERSION

FPS = 30.0


@pytest.fixture
def golden_motion():
    qpos = np.load(DATA_DIR / "golden_qpos_unitree_g1.npy")
    joint_names = [f"joint_{i}" for i in range(qpos.shape[1] - 7)]
    return qpos[:, :3], qpos[:, 3:7], qpos[:, 7:], joint_names


def test_npz_roundtrip(tmp_path, golden_motion):
    root_pos, root_rot_wxyz, dof_pos, joint_names = golden_motion
    path = tmp_path / "motion.npz"
    save_robot_motion(path, FPS, root_pos, root_rot_wxyz, dof_pos, joint_names)

    motion_data, fps, loaded_pos, loaded_rot, loaded_dof, local, links = (
        load_robot_motion(str(path))
    )
    assert fps == FPS
    np.testing.assert_array_equal(loaded_pos, root_pos)
    np.testing.assert_array_equal(loaded_rot, root_rot_wxyz)
    np.testing.assert_array_equal(loaded_dof, dof_pos)
    assert list(motion_data["joint_names"]) == joint_names
    assert int(motion_data["schema_version"]) == SCHEMA_VERSION
    assert local is None and links is None


def test_legacy_pickle_still_loads(tmp_path, golden_motion):
    root_pos, root_rot_wxyz, dof_pos, _ = golden_motion
    path = tmp_path / "motion.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "fps": FPS,
                "root_pos": root_pos,
                "root_rot": root_rot_wxyz[:, [1, 2, 3, 0]],  # legacy xyzw
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            },
            f,
        )
    _, fps, loaded_pos, loaded_rot, loaded_dof, _, _ = load_robot_motion(str(path))
    assert fps == FPS
    np.testing.assert_array_equal(loaded_pos, root_pos)
    # The legacy loader converts back to wxyz.
    np.testing.assert_array_equal(loaded_rot, root_rot_wxyz)
    np.testing.assert_array_equal(loaded_dof, dof_pos)


def test_save_rejects_unnormalized_quaternions(tmp_path, golden_motion):
    root_pos, root_rot_wxyz, dof_pos, joint_names = golden_motion
    bad_rot = root_rot_wxyz * 2.0
    with pytest.raises(ValueError, match="not normalized"):
        save_robot_motion(
            tmp_path / "m.npz", FPS, root_pos, bad_rot, dof_pos, joint_names
        )


def test_save_rejects_mismatched_joint_names(tmp_path, golden_motion):
    root_pos, root_rot_wxyz, dof_pos, joint_names = golden_motion
    with pytest.raises(ValueError, match="joint names"):
        save_robot_motion(
            tmp_path / "m.npz", FPS, root_pos, root_rot_wxyz, dof_pos, joint_names[:-1]
        )


def test_save_rejects_non_finite(tmp_path, golden_motion):
    root_pos, root_rot_wxyz, dof_pos, joint_names = golden_motion
    dof_pos = dof_pos.copy()
    dof_pos[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_robot_motion(
            tmp_path / "m.npz", FPS, root_pos, root_rot_wxyz, dof_pos, joint_names
        )


def test_load_rejects_newer_schema(tmp_path, golden_motion):
    root_pos, root_rot_wxyz, dof_pos, joint_names = golden_motion
    path = tmp_path / "motion.npz"
    save_robot_motion(path, FPS, root_pos, root_rot_wxyz, dof_pos, joint_names)
    data = dict(np.load(path))
    data["schema_version"] = np.array(SCHEMA_VERSION + 1)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="newer than the supported version"):
        load_robot_motion(str(path))
