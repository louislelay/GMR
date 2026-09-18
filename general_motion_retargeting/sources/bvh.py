"""BVH motion loading."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from ..models import FloatArray, HumanFrame

_WORLD_FROM_BVH = Rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
)


@dataclass(frozen=True)
class BvhMotionData:
    """BVH frames and source timing."""

    frames: tuple[HumanFrame, ...]
    fps: float
    height: float
    source_fps: float


@dataclass(frozen=True)
class _Hierarchy:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    offsets: FloatArray
    channels: tuple[tuple[str, ...], ...]


def _parse_hierarchy(lines: list[str]) -> _Hierarchy:
    names: list[str] = []
    parents: list[int] = []
    offsets: list[tuple[float, float, float]] = []
    channels: list[tuple[str, ...]] = []
    active = -1
    in_end_site = False

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith(("ROOT ", "JOINT ")):
            names.append(line.split(maxsplit=1)[1])
            parents.append(active)
            offsets.append((0.0, 0.0, 0.0))
            channels.append(())
            active = len(names) - 1
            continue
        if line == "End Site":
            in_end_site = True
            continue
        if line == "}":
            if in_end_site:
                in_end_site = False
                continue
            active = parents[active]
            continue
        if in_end_site or active < 0:
            continue
        if line.startswith("OFFSET "):
            values = tuple(float(value) for value in line.split()[1:])
            if len(values) != 3:
                raise ValueError(f"invalid BVH offset: {line}")
            offsets[active] = values
            continue
        if line.startswith("CHANNELS "):
            parts = line.split()
            count = int(parts[1])
            joint_channels = tuple(parts[2:])
            if len(joint_channels) != count:
                raise ValueError(f"invalid BVH channels: {line}")
            channels[active] = joint_channels

    if not names:
        raise ValueError("BVH hierarchy contains no joints")
    return _Hierarchy(
        names=tuple(names),
        parents=tuple(parents),
        offsets=np.asarray(offsets, dtype=np.float64),
        channels=tuple(channels),
    )


def _motion_rows(lines: list[str]) -> tuple[FloatArray, float]:
    frame_count: int | None = None
    frame_time: float | None = None
    first_row: int | None = None
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("Frames:"):
            frame_count = int(line.split(":", maxsplit=1)[1])
        if line.startswith("Frame Time:"):
            frame_time = float(line.split(":", maxsplit=1)[1])
            first_row = index + 1
            break
    if frame_count is None or frame_time is None or first_row is None:
        raise ValueError("BVH motion header is incomplete")
    rows = [
        [float(value) for value in line.split()]
        for line in lines[first_row : first_row + frame_count]
        if line.strip()
    ]
    if len(rows) != frame_count:
        raise ValueError(f"BVH declares {frame_count} frames but contains {len(rows)}")
    return np.asarray(rows, dtype=np.float64), 1.0 / frame_time


def _local_poses(
    hierarchy: _Hierarchy, rows: FloatArray
) -> tuple[FloatArray, FloatArray]:
    frame_count = rows.shape[0]
    positions = np.broadcast_to(
        hierarchy.offsets,
        (frame_count, *hierarchy.offsets.shape),
    ).copy()
    quaternions = np.zeros((frame_count, len(hierarchy.names), 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    column = 0

    for joint, channels in enumerate(hierarchy.channels):
        values = rows[:, column : column + len(channels)]
        column += len(channels)
        rotation_columns: list[int] = []
        rotation_order = ""
        for channel, name in enumerate(channels):
            axis = "XYZ".index(name[0])
            if name.endswith("position"):
                positions[:, joint, axis] += values[:, channel]
                continue
            if name.endswith("rotation"):
                rotation_columns.append(channel)
                rotation_order += name[0]
                continue
            raise ValueError(f"unsupported BVH channel: {name}")
        if rotation_columns:
            quaternions[:, joint] = Rotation.from_euler(
                rotation_order,
                values[:, rotation_columns],
                degrees=True,
            ).as_quat(scalar_first=True)

    if column != rows.shape[1]:
        raise ValueError(
            f"BVH frames contain {rows.shape[1]} values; hierarchy declares {column}"
        )
    return positions, quaternions


def _global_poses(
    hierarchy: _Hierarchy,
    local_positions: FloatArray,
    local_quaternions: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    global_positions = np.empty_like(local_positions)
    global_quaternions = np.empty_like(local_quaternions)
    for joint, parent in enumerate(hierarchy.parents):
        local_rotation = Rotation.from_quat(
            local_quaternions[:, joint],
            scalar_first=True,
        )
        if parent < 0:
            global_positions[:, joint] = local_positions[:, joint]
            global_quaternions[:, joint] = local_quaternions[:, joint]
            continue
        parent_rotation = Rotation.from_quat(
            global_quaternions[:, parent],
            scalar_first=True,
        )
        global_positions[:, joint] = global_positions[
            :, parent
        ] + parent_rotation.apply(local_positions[:, joint])
        global_quaternions[:, joint] = (parent_rotation * local_rotation).as_quat(
            scalar_first=True
        )
    return global_positions, global_quaternions


def _resample(
    positions: FloatArray,
    quaternions: FloatArray,
    source_fps: float,
    target_fps: float,
) -> tuple[FloatArray, FloatArray, float]:
    frame_count = positions.shape[0]
    target_count = max(1, round(frame_count * target_fps / source_fps))
    samples = np.linspace(0.0, frame_count - 1, target_count)
    aligned_fps = target_count / frame_count * source_fps
    if frame_count == 1:
        return (
            np.repeat(positions, target_count, axis=0),
            np.repeat(quaternions, target_count, axis=0),
            aligned_fps,
        )
    resampled_positions = interp1d(
        np.arange(frame_count),
        positions,
        axis=0,
    )(samples)
    resampled_quaternions = np.stack(
        [
            Slerp(
                np.arange(frame_count),
                Rotation.from_quat(quaternions[:, joint], scalar_first=True),
            )(samples).as_quat(scalar_first=True)
            for joint in range(quaternions.shape[1])
        ],
        axis=1,
    )
    return resampled_positions, resampled_quaternions, aligned_fps


def load_bvh_motion(
    path: Path,
    *,
    convention: str = "lafan1",
    target_fps: float,
) -> BvhMotionData:
    """Load and resample a LAFAN1 or Nokov BVH motion."""
    if convention not in {"lafan1", "nokov"}:
        raise ValueError(f"unsupported BVH convention: {convention}")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target frame rate must be positive and finite")

    lines = path.read_text(encoding="utf-8").splitlines()
    motion_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "MOTION"),
        None,
    )
    if motion_index is None:
        raise ValueError("BVH file has no MOTION section")
    hierarchy = _parse_hierarchy(lines[:motion_index])
    rows, source_fps = _motion_rows(lines[motion_index + 1 :])
    expected_columns = sum(len(channels) for channels in hierarchy.channels)
    if rows.shape[1] != expected_columns:
        raise ValueError(
            f"BVH frames contain {rows.shape[1]} values; "
            f"hierarchy declares {expected_columns}"
        )

    local_positions, local_quaternions = _local_poses(hierarchy, rows)
    positions, quaternions = _global_poses(
        hierarchy,
        local_positions,
        local_quaternions,
    )
    positions, quaternions, aligned_fps = _resample(
        positions,
        quaternions,
        source_fps,
        target_fps,
    )
    positions = _WORLD_FROM_BVH.apply(positions.reshape(-1, 3)).reshape(positions.shape)
    positions /= 100.0
    quaternions = (
        (
            _WORLD_FROM_BVH
            * Rotation.from_quat(quaternions.reshape(-1, 4), scalar_first=True)
        )
        .as_quat(scalar_first=True)
        .reshape(quaternions.shape)
    )

    left_toe = "LeftToe" if convention == "lafan1" else "LeftToeBase"
    right_toe = "RightToe" if convention == "lafan1" else "RightToeBase"
    name_to_index = {name: index for index, name in enumerate(hierarchy.names)}
    required = {"LeftFoot", "RightFoot", left_toe, right_toe}
    missing = required.difference(name_to_index)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"BVH is missing required foot joints: {names}")

    frames: list[HumanFrame] = []
    for frame_index in range(len(positions)):
        frame = {
            name: (
                positions[frame_index, joint],
                quaternions[frame_index, joint],
            )
            for joint, name in enumerate(hierarchy.names)
        }
        frame["LeftFootMod"] = (
            frame["LeftFoot"][0],
            frame[left_toe][1],
        )
        frame["RightFootMod"] = (
            frame["RightFoot"][0],
            frame[right_toe][1],
        )
        frames.append(frame)

    return BvhMotionData(
        frames=tuple(frames),
        fps=aligned_fps,
        height=1.75,
        source_fps=source_fps,
    )
