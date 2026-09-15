"""Load immutable versioned retargeting profiles."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import FrameTransform, HumanFrame, Match, RetargetingProfile

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


class _FrameData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    body: str = Field(min_length=1)
    position: Vector3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (1.0, 0.0, 0.0, 0.0)


class _MatchData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    robot: _FrameData
    human: _FrameData
    position_weight: float = Field(ge=0.0)
    orientation_weight: float = Field(ge=0.0)


class _ProfileData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    identifier: str | None = None
    robot_root_name: str
    human_root_name: str
    ground_height: float
    human_height_assumption: float = Field(gt=0.0)
    human_scale_table: dict[str, float]
    stages: tuple[dict[str, _MatchData], ...]


def _immutable_array(values: Sequence[float], *, size: int, where: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{where} must contain {size} values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{where} contains non-finite values")
    array.setflags(write=False)
    return array


def _frame(value: _FrameData, *, where: str) -> FrameTransform:
    position = _immutable_array(value.position, size=3, where=f"{where} position")
    rotation = _immutable_array(value.rotation, size=4, where=f"{where} rotation")
    norm = float(np.linalg.norm(rotation))
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        raise ValueError(f"{where} rotation must be a unit wxyz quaternion")
    return FrameTransform(body=value.body, position=position, rotation=rotation)


def _match(identifier: str, value: _MatchData) -> Match:
    weights = (value.position_weight, value.orientation_weight)
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError(f"match {identifier!r} has non-finite weights")
    return Match(
        identifier=identifier,
        robot=_frame(value.robot, where=f"match {identifier!r} robot"),
        human=_frame(value.human, where=f"match {identifier!r} human"),
        position_weight=value.position_weight,
        orientation_weight=value.orientation_weight,
    )


def _matches(values: Mapping[str, _MatchData]) -> tuple[Match, ...]:
    return tuple(_match(identifier, value) for identifier, value in values.items())


def _validate_common(
    *,
    path: Path,
    human_root: str,
    ground_height: float,
    scales: Mapping[str, float],
    stages: tuple[tuple[Match, ...], ...],
) -> None:
    if not math.isfinite(ground_height):
        raise ValueError(f"{path}: ground_height must be finite")
    if human_root not in scales:
        raise ValueError(f"{path}: human root {human_root!r} is missing from scales")
    invalid_scales = {
        name: scale
        for name, scale in scales.items()
        if not math.isfinite(scale) or scale <= 0.0
    }
    if invalid_scales:
        raise ValueError(f"{path}: human scales must be positive and finite")
    if not stages or not any(stages):
        raise ValueError(f"{path}: profile must contain an enabled non-empty stage")
    missing_scales = {
        match.human.body
        for stage in stages
        for match in stage
        if match.position_weight != 0.0 or match.orientation_weight != 0.0
    }.difference(scales)
    if missing_scales:
        names = ", ".join(sorted(missing_scales))
        raise ValueError(f"{path}: active human bodies missing from scales: {names}")


def load_profile(path: Path, *, source_format: str, robot: str) -> RetargetingProfile:
    """Load and validate a retargeting profile.

    Args:
        path: JSON profile path.
        source_format: Human-frame convention produced by the source adapter.
        robot: Identifier of the target robot.

    Returns:
        Immutable validated retargeting profile.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = _ProfileData.model_validate(raw)
    stages = tuple(_matches(stage) for stage in data.stages)
    identifier = data.identifier or f"{source_format}_to_{robot}"

    scales = MappingProxyType(dict(data.human_scale_table))
    _validate_common(
        path=path,
        human_root=data.human_root_name,
        ground_height=data.ground_height,
        scales=scales,
        stages=stages,
    )
    return RetargetingProfile(
        identifier=identifier,
        source_format=source_format,
        robot=robot,
        robot_root=data.robot_root_name,
        human_root=data.human_root_name,
        ground_height=data.ground_height,
        human_height_assumption=data.human_height_assumption,
        human_scales=scales,
        stages=stages,
    )


def validate_human_frame(frame: HumanFrame, required_bodies: frozenset[str]) -> None:
    """Validate names and pose arrays in one human frame.

    Args:
        frame: Mapping from body names to position and wxyz orientation.
        required_bodies: Bodies required by the active profile.
    """
    missing = required_bodies.difference(frame)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"human frame is missing required bodies: {names}")
    for name in required_bodies:
        position, rotation = frame[name]
        _immutable_array(position, size=3, where=f"human body {name!r} position")
        quaternion = _immutable_array(
            rotation, size=4, where=f"human body {name!r} rotation"
        )
        if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-3):
            raise ValueError(
                f"human body {name!r} rotation must be a unit wxyz quaternion"
            )
