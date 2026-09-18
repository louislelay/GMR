"""SMPL-X motion loading."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from smplx.joint_names import JOINT_NAMES

from ..models import FloatArray, HumanFrame, IntArray


@dataclass(frozen=True)
class SmplxMotionData:
    """SMPL-X frames and visualization data."""

    frames: tuple[HumanFrame, ...]
    fps: float
    height: float
    source_fps: float
    vertices: FloatArray
    faces: IntArray


def _normalize_gender(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode()
    gender = str(value).lower()
    if gender not in {"neutral", "male", "female"}:
        raise ValueError(f"unsupported SMPL-X gender: {gender!r}")
    return gender


def detect_body_model_ext(body_models: str | os.PathLike[str], gender: str) -> str:
    """Return the available SMPL-X body-model extension."""
    model_dir = Path(body_models) / "smplx"
    candidates = tuple(
        model_dir / f"SMPLX_{gender.upper()}.{extension}"
        for extension in ("npz", "pkl")
    )
    for path in candidates:
        if path.exists():
            return path.suffix.removeprefix(".")
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"no SMPL-X body model found; expected one of: {expected}")


def _sample_positions(
    frame_count: int, source_fps: float, target_fps: float
) -> tuple[FloatArray, float]:
    if frame_count <= 0:
        raise ValueError("SMPL-X motion must contain at least one frame")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("SMPL-X frame rate must be positive and finite")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target frame rate must be positive and finite")
    target_count = max(1, round(frame_count * target_fps / source_fps))
    positions = np.linspace(0.0, frame_count - 1, target_count, dtype=np.float64)
    aligned_fps = target_count / frame_count * source_fps
    return positions, aligned_fps


def _resample_rotvecs(values: FloatArray, positions: FloatArray) -> FloatArray:
    frame_count = values.shape[0]
    if frame_count == 1:
        return np.repeat(values, len(positions), axis=0)
    samples = tuple(
        Slerp(
            np.arange(frame_count),
            Rotation.from_rotvec(values[:, joint]),
        )(positions).as_rotvec()
        for joint in range(values.shape[1])
    )
    return np.stack(samples, axis=1)


def _resample_vectors(values: FloatArray, positions: FloatArray) -> FloatArray:
    if values.shape[0] == 1:
        return np.repeat(values, len(positions), axis=0)
    return np.asarray(
        interp1d(np.arange(values.shape[0]), values, axis=0)(positions),
        dtype=np.float64,
    )


def _global_frames(
    global_orient: FloatArray,
    body_pose: FloatArray,
    joints: FloatArray,
    parents: np.ndarray,
) -> tuple[HumanFrame, ...]:
    names = JOINT_NAMES[: len(parents)]
    frames: list[HumanFrame] = []
    for frame_index, root_rotvec in enumerate(global_orient):
        orientations: list[Rotation] = []
        frame: dict[str, tuple[FloatArray, FloatArray]] = {}
        for joint_index, name in enumerate(names):
            local = (
                Rotation.from_rotvec(root_rotvec)
                if joint_index == 0
                else Rotation.from_rotvec(body_pose[frame_index, joint_index])
            )
            rotation = (
                local
                if joint_index == 0
                else orientations[int(parents[joint_index])] * local
            )
            orientations.append(rotation)
            frame[name] = (
                np.asarray(joints[frame_index, joint_index], dtype=np.float64),
                rotation.as_quat(scalar_first=True),
            )
        frames.append(frame)
    return tuple(frames)


def load_smplx_motion(
    path: Path, body_models: Path, *, target_fps: float
) -> SmplxMotionData:
    """Load and resample one SMPL-X NPZ motion."""
    with np.load(path, allow_pickle=True) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    gender = _normalize_gender(data["gender"])
    model = smplx.create(
        str(body_models),
        "smplx",
        gender=gender,
        use_pca=False,
        ext=detect_body_model_ext(body_models, gender),
    )
    frame_count = int(data["pose_body"].shape[0])
    output = model(
        betas=torch.as_tensor(data["betas"]).float().reshape(1, -1),  # (1, 16)
        global_orient=torch.as_tensor(data["root_orient"]).float(),  # (N, 3)
        body_pose=torch.as_tensor(data["pose_body"]).float(),  # (N, 63)
        transl=torch.as_tensor(data["trans"]).float(),  # (N, 3)
        left_hand_pose=torch.zeros(frame_count, 45),
        right_hand_pose=torch.zeros(frame_count, 45),
        jaw_pose=torch.zeros(frame_count, 3),
        leye_pose=torch.zeros(frame_count, 3),
        reye_pose=torch.zeros(frame_count, 3),
        return_full_pose=True,
    )

    source_fps = float(data["mocap_frame_rate"].item())
    positions, aligned_fps = _sample_positions(frame_count, source_fps, target_fps)
    global_orient = _resample_rotvecs(
        np.asarray(output.global_orient.detach().cpu(), dtype=np.float64)[:, None, :],
        positions,
    )[:, 0]
    full_pose = _resample_rotvecs(
        np.asarray(output.full_pose.detach().cpu(), dtype=np.float64).reshape(
            frame_count, -1, 3
        ),
        positions,
    )
    joints = _resample_vectors(
        np.asarray(output.joints.detach().cpu(), dtype=np.float64),
        positions,
    )
    vertices = _resample_vectors(
        np.asarray(output.vertices.detach().cpu(), dtype=np.float64),
        positions,
    )
    frames = _global_frames(
        global_orient,
        full_pose,
        joints,
        np.asarray(model.parents),
    )
    height = 1.66 + 0.1 * float(data["betas"].reshape(-1)[0])
    return SmplxMotionData(
        frames=frames,
        fps=aligned_fps,
        height=height,
        source_fps=source_fps,
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(model.faces, dtype=np.int64),
    )
