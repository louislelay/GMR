"""Versioned, pickle-free persistence for canonical robot motions."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .models import RobotMotion

SCHEMA_VERSION = 1
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "robot",
        "source_format",
        "profile",
        "fps",
        "start_time",
        "source_identifier",
        "source_fps",
        "root_positions",
        "root_quaternions",
        "joint_positions",
        "joint_names",
    }
)


def save_robot_motion(path: Path, motion: RobotMotion) -> None:
    """Atomically save a canonical robot motion as compressed NPZ.

    Args:
        path: Destination ending in ``.npz``.
        motion: Validated robot motion.
    """
    if path.suffix.lower() != ".npz":
        raise ValueError("robot motion path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    source_identifier = motion.source_identifier or ""
    source_fps = motion.source_fps if motion.source_fps is not None else np.nan
    with NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
            robot=np.asarray(motion.robot),
            source_format=np.asarray(motion.source_format),
            profile=np.asarray(motion.profile),
            fps=np.asarray(motion.fps, dtype=np.float64),
            start_time=np.asarray(motion.start_time, dtype=np.float64),
            source_identifier=np.asarray(source_identifier),
            source_fps=np.asarray(source_fps, dtype=np.float64),
            root_positions=motion.root_positions,
            root_quaternions=motion.root_quaternions,
            joint_positions=motion.joint_positions,
            joint_names=np.asarray(motion.joint_names),
        )
    temporary_path.replace(path)


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar")
    return value.item()


def load_robot_motion(path: Path) -> RobotMotion:
    """Load and validate a canonical robot-motion NPZ.

    Args:
        path: Source NPZ path.

    Returns:
        Immutable validated robot motion.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = _REQUIRED_KEYS.difference(data.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"robot motion is missing fields: {names}")
        version = int(_scalar(data, "schema_version"))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported robot-motion schema {version}; expected {SCHEMA_VERSION}"
            )
        source_identifier = str(_scalar(data, "source_identifier")) or None
        source_fps_value = float(_scalar(data, "source_fps"))
        source_fps = None if np.isnan(source_fps_value) else source_fps_value
        return RobotMotion(
            robot=str(_scalar(data, "robot")),
            source_format=str(_scalar(data, "source_format")),
            profile=str(_scalar(data, "profile")),
            fps=float(_scalar(data, "fps")),
            start_time=float(_scalar(data, "start_time")),
            source_identifier=source_identifier,
            source_fps=source_fps,
            root_positions=np.asarray(data["root_positions"]),
            root_quaternions=np.asarray(data["root_quaternions"]),
            joint_positions=np.asarray(data["joint_positions"]),
            joint_names=tuple(str(name) for name in data["joint_names"]),
        )
