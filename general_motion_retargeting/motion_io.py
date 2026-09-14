"""Versioned, pickle-free persistence for retargeted robot motions.

The NPZ schema stores explicit field names and conventions so files remain
loadable across library versions:

- schema_version: int, currently 1
- fps: float
- joint_names: array of str, one per dof_pos column, in qpos order
- root_pos: (num_frames, 3) float
- root_rot_wxyz: (num_frames, 4) float, scalar-first unit quaternions
- dof_pos: (num_frames, num_joints) float

Legacy pickle files (.pkl, xyzw root rotation, no joint names) remain
loadable through load_robot_motion.
"""

import mujoco as mj
import numpy as np

from .data_loader import load_robot_motion as _load_legacy_pickle

SCHEMA_VERSION = 1


def robot_joint_names(model: mj.MjModel) -> list[str]:
    """Names of the robot's non-free joints, one per dof_pos column."""
    return [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_FREE
    ]


def save_robot_motion(
    path,
    fps: float,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    joint_names: list[str],
) -> None:
    """Save a robot motion to a versioned NPZ archive.

    Args:
        path: Output path, conventionally ending in .npz.
        fps: Frame rate of the motion.
        root_pos: (num_frames, 3) floating-base positions.
        root_rot_wxyz: (num_frames, 4) scalar-first unit quaternions.
        dof_pos: (num_frames, num_joints) joint positions.
        joint_names: One name per dof_pos column, in qpos order.

    Raises:
        ValueError: If shapes are inconsistent, values are non-finite, or the
            quaternions are not normalized.
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot_wxyz = np.asarray(root_rot_wxyz, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)

    num_frames = root_pos.shape[0]
    if root_pos.shape != (num_frames, 3):
        raise ValueError(f"root_pos must have shape (N, 3), got {root_pos.shape}")
    if root_rot_wxyz.shape != (num_frames, 4):
        raise ValueError(
            f"root_rot_wxyz must have shape ({num_frames}, 4), "
            f"got {root_rot_wxyz.shape}"
        )
    if dof_pos.shape[0] != num_frames:
        raise ValueError(
            f"dof_pos has {dof_pos.shape[0]} frames, expected {num_frames}"
        )
    if dof_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"dof_pos has {dof_pos.shape[1]} joints but {len(joint_names)} "
            "joint names were given"
        )
    for name, array in (
        ("root_pos", root_pos),
        ("root_rot_wxyz", root_rot_wxyz),
        ("dof_pos", dof_pos),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(root_rot_wxyz, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("root_rot_wxyz quaternions are not normalized")

    np.savez_compressed(
        path,
        schema_version=SCHEMA_VERSION,
        fps=float(fps),
        joint_names=np.array(joint_names, dtype=np.str_),
        root_pos=root_pos,
        root_rot_wxyz=root_rot_wxyz,
        dof_pos=dof_pos,
    )


def load_robot_motion(motion_file):
    """Load a robot motion from an NPZ archive or a legacy pickle.

    Files ending in .npz are read with the schema documented in this module;
    anything else falls back to the legacy pickle format.

    Returns:
        Tuple of (motion_data, fps, root_pos, root_rot_wxyz, dof_pos,
        local_body_pos, link_body_list), matching the legacy loader. For NPZ
        files, motion_data is the dict of stored arrays and the last two
        elements are None.
    """
    if not str(motion_file).endswith(".npz"):
        return _load_legacy_pickle(motion_file)

    with np.load(motion_file) as archive:
        version = int(archive["schema_version"])
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{motion_file}: schema version {version} is newer than the "
                f"supported version {SCHEMA_VERSION}; update the library"
            )
        motion_data = {key: archive[key] for key in archive.files}

    return (
        motion_data,
        float(motion_data["fps"]),
        motion_data["root_pos"],
        motion_data["root_rot_wxyz"],
        motion_data["dof_pos"],
        None,
        None,
    )
