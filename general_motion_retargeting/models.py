"""Typed domain models shared by GMR applications and extensions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import mujoco as mj

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
HumanFrame: TypeAlias = Mapping[str, tuple[FloatArray, FloatArray]]


def _readonly_float_array(value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _readonly_int_array(value: npt.ArrayLike) -> IntArray:
    array = np.asarray(value, dtype=np.int64).copy()
    array.setflags(write=False)
    return array


def _readonly_frame(frame: HumanFrame) -> HumanFrame:
    return MappingProxyType(
        {
            name: (
                _readonly_float_array(position),
                _readonly_float_array(rotation),
            )
            for name, (position, rotation) in frame.items()
        }
    )


@dataclass(frozen=True)
class HumanMotion:
    """Sequence of global human body poses."""

    frames: tuple[HumanFrame, ...]
    fps: float
    height: float
    source_format: str
    source_identifier: str | None = None
    source_fps: float | None = None
    body_vertices: FloatArray | None = None
    body_faces: IntArray | None = None

    def __post_init__(self) -> None:
        """Validate the motion and freeze owned arrays."""
        if not self.frames:
            raise ValueError("human motion must contain at least one frame")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be positive and finite")
        if not np.isfinite(self.height) or self.height <= 0.0:
            raise ValueError("height must be positive and finite")
        if self.source_fps is not None and (
            not np.isfinite(self.source_fps) or self.source_fps <= 0.0
        ):
            raise ValueError("source_fps must be positive and finite")
        frames = tuple(_readonly_frame(frame) for frame in self.frames)
        object.__setattr__(self, "frames", frames)

        if (self.body_vertices is None) != (self.body_faces is None):
            raise ValueError("body_vertices and body_faces must be provided together")
        if self.body_vertices is None or self.body_faces is None:
            return
        vertices = _readonly_float_array(self.body_vertices)
        faces = _readonly_int_array(self.body_faces)
        if vertices.ndim != 3 or vertices.shape[0] != len(frames):
            raise ValueError("body_vertices must have shape (frames, vertices, 3)")
        if vertices.shape[2] != 3:
            raise ValueError("body_vertices must have shape (frames, vertices, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("body_faces must have shape (faces, 3)")
        object.__setattr__(self, "body_vertices", vertices)
        object.__setattr__(self, "body_faces", faces)

    @property
    def frame_count(self) -> int:
        """Number of motion frames."""
        return len(self.frames)

    @property
    def duration(self) -> float:
        """Motion duration in seconds."""
        return self.frame_count / self.fps


@dataclass(frozen=True)
class RobotMotion:
    """Retargeted robot trajectory in a named scalar-qpos layout."""

    robot: str
    source_format: str
    profile: str
    fps: float
    root_positions: FloatArray
    root_quaternions: FloatArray
    joint_positions: FloatArray
    joint_names: tuple[str, ...]
    start_time: float = 0.0
    source_identifier: str | None = None
    source_fps: float | None = None

    def __post_init__(self) -> None:
        """Validate the trajectory and freeze owned NumPy buffers."""
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be positive and finite")
        if not np.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if self.source_fps is not None and (
            not np.isfinite(self.source_fps) or self.source_fps <= 0.0
        ):
            raise ValueError("source_fps must be positive and finite")
        root_positions = _readonly_float_array(self.root_positions)
        root_quaternions = _readonly_float_array(self.root_quaternions)
        joint_positions = _readonly_float_array(self.joint_positions)
        frame_count = root_positions.shape[0]
        expected_shapes = (
            (root_positions.shape, (frame_count, 3), "root_positions"),
            (root_quaternions.shape, (frame_count, 4), "root_quaternions"),
            (
                joint_positions.shape,
                (frame_count, len(self.joint_names)),
                "joint_positions",
            ),
        )
        for actual, expected, name in expected_shapes:
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")
        if frame_count == 0:
            raise ValueError("robot motion must contain at least one frame")
        if not all(
            np.isfinite(array).all()
            for array in (root_positions, root_quaternions, joint_positions)
        ):
            raise ValueError("robot motion arrays must contain only finite values")
        norms = np.linalg.norm(root_quaternions, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-3):
            raise ValueError("root_quaternions must be unit wxyz quaternions")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "root_positions", root_positions)
        object.__setattr__(self, "root_quaternions", root_quaternions)
        object.__setattr__(self, "joint_positions", joint_positions)

    @property
    def frame_count(self) -> int:
        """Number of trajectory frames."""
        return self.root_positions.shape[0]

    @property
    def duration(self) -> float:
        """Motion duration in seconds."""
        return self.frame_count / self.fps

    @property
    def qpos(self) -> FloatArray:
        """Return a new MuJoCo-style qpos matrix."""
        return np.concatenate(
            (self.root_positions, self.root_quaternions, self.joint_positions),
            axis=1,
        )

    def trim(self, start: float, end: float) -> RobotMotion:
        """Return the motion between two times in seconds."""
        if not 0.0 <= start < end <= self.duration:
            raise ValueError(
                f"trim bounds must satisfy 0 <= start < end <= {self.duration}"
            )
        first = int(np.floor(start * self.fps))
        last = min(int(np.ceil(end * self.fps)), self.frame_count)
        return RobotMotion(
            robot=self.robot,
            source_format=self.source_format,
            profile=self.profile,
            fps=self.fps,
            root_positions=self.root_positions[first:last],
            root_quaternions=self.root_quaternions[first:last],
            joint_positions=self.joint_positions[first:last],
            joint_names=self.joint_names,
            start_time=self.start_time + first / self.fps,
            source_identifier=self.source_identifier,
            source_fps=self.source_fps,
        )


@dataclass(frozen=True)
class FrameTransform:
    """Named body frame with a local position and wxyz orientation."""

    body: str
    position: FloatArray
    rotation: FloatArray


@dataclass(frozen=True)
class Match:
    """One human-frame to robot-frame IK objective."""

    identifier: str
    robot: FrameTransform
    human: FrameTransform
    position_weight: float
    orientation_weight: float


@dataclass(frozen=True)
class RetargetingProfile:
    """Validated source-to-robot retargeting parameters."""

    identifier: str
    source_format: str
    robot: str
    robot_root: str
    human_root: str
    ground_height: float
    human_height_assumption: float
    human_scales: Mapping[str, float]
    stages: tuple[tuple[Match, ...], ...]


class RobotAssets(Protocol):
    """Provider of editable robot and visualization specifications."""

    def load_model_spec(self) -> mj.MjSpec:
        """Load the robot-only specification."""
        ...

    def load_scene_spec(self) -> mj.MjSpec:
        """Load the visualization-scene specification."""
        ...


@dataclass(frozen=True)
class TrackingCamera:
    """Camera configuration tracking a robot body."""

    body: str
    distance: float
    azimuth: float = 135.0
    elevation: float = -10.0


@dataclass(frozen=True)
class RobotSpec:
    """Robot identity, assets, root frame, and default camera."""

    identifier: str
    assets: RobotAssets
    root_body: str
    camera: TrackingCamera


@dataclass(frozen=True)
class SolverSettings:
    """Differential-IK solver and convergence settings."""

    solver: str = "daqp"
    damping: float = 5e-1
    max_iterations: int = 10
    improvement_threshold: float = 1e-3
    use_velocity_limits: bool = False
    velocity_limit: float = 3.0 * np.pi
