"""Typed domain models shared by GMR applications and extensions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import mujoco as mj
import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
HumanFrame: TypeAlias = Mapping[str, tuple[FloatArray, FloatArray]]


@dataclass(frozen=True)
class HumanMotion:
    """Sequence of global human body poses."""

    frames: tuple[HumanFrame, ...]
    fps: float
    height: float
    source_format: str


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
