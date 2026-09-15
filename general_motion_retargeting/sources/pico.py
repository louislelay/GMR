"""Lazy PICO body tracking through the separately installed SDK binding."""

import time
from importlib import import_module
from types import ModuleType

import numpy as np
from scipy.spatial.transform import Rotation

from ..streaming import HumanFrameSample

_BODY_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
_UNITY_TO_WORLD = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_UNITY_ROTATION = Rotation.from_matrix(_UNITY_TO_WORLD)


class PicoSource:
    """PICO live source using the optional XRoboToolkit Python binding."""

    def __init__(self, *, fps: float = 60.0) -> None:
        """Configure the nominal frame rate."""
        self._fps = fps
        self._sdk: ModuleType | None = None
        self._last_timestamp = -1
        self._sequence = 0

    @property
    def source_format(self) -> str:
        """Profile source convention."""
        return "xrobot"

    @property
    def fps(self) -> float:
        """Nominal source frame rate."""
        return self._fps

    def open(self) -> None:
        """Lazily import and initialize the separately installed SDK."""
        try:
            sdk = import_module("xrobotoolkit_sdk")
        except ImportError as error:
            raise ValueError(
                "PICO streaming requires the separately installed "
                "xrobotoolkit_sdk binding"
            ) from error
        sdk.init()
        self._sdk = sdk

    def read(self, timeout: float | None = None) -> HumanFrameSample | None:
        """Read the newest PICO frame without duplicating SDK timestamps."""
        if self._sdk is None:
            raise RuntimeError("PICO source is not open")
        if not self._sdk.is_body_data_available():
            time.sleep(min(timeout or 0.001, 0.001))
            return None
        timestamp = int(self._sdk.get_body_timestamp_ns())
        if timestamp == self._last_timestamp:
            time.sleep(min(timeout or 0.001, 0.001))
            return None
        self._last_timestamp = timestamp
        poses = self._sdk.get_body_joints_pose()
        frame = {}
        for index, name in enumerate(_BODY_NAMES):
            values = np.asarray(poses[index], dtype=np.float64)
            position = _UNITY_TO_WORLD @ values[:3]
            rotation = _UNITY_ROTATION * Rotation.from_quat(values[3:7])
            frame[name] = (
                position,
                rotation.as_quat(scalar_first=True),
            )
        sample = HumanFrameSample(
            sequence=self._sequence,
            frame=frame,
            source_timestamp=timestamp / 1e9,
            received_monotonic=time.monotonic(),
        )
        self._sequence += 1
        return sample

    def close(self) -> None:
        """Release this adapter's SDK reference."""
        self._sdk = None
