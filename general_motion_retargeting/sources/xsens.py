"""Xsens MVN UDP live source."""

import time
from importlib import import_module
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import HumanFrame
from ..streaming import HumanFrameSample


class _Device(Protocol):
    def init(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_sample_counter(self) -> int: ...

    def get_link_names(self) -> list[str]: ...

    def get_link_position(self, name: str) -> np.ndarray: ...

    def get_link_orientation(self, name: str) -> np.ndarray: ...


_LINK_NAMES = {
    "pelvis": "Pelvis",
    "l5": "Spine",
    "l3": "Spine1",
    "t12": "Spine2",
    "t8": "Chest",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "Left_Shoulder",
    "left_upper_arm": "Left_UpperArm",
    "left_forearm": "Left_Forearm",
    "left_hand": "Left_Hand",
    "right_shoulder": "Right_Shoulder",
    "right_upper_arm": "Right_UpperArm",
    "right_forearm": "Right_Forearm",
    "right_hand": "Right_Hand",
    "left_upper_leg": "Left_UpperLeg",
    "left_lower_leg": "Left_LowerLeg",
    "left_foot": "Left_Foot",
    "left_toe": "Left_Toe",
    "right_upper_leg": "Right_UpperLeg",
    "right_lower_leg": "Right_LowerLeg",
    "right_foot": "Right_Foot",
    "right_toe": "Right_Toe",
}
_REQUIRED_BODIES = {
    "Pelvis",
    "Chest",
    "Left_UpperArm",
    "Left_Forearm",
    "Left_Hand",
    "Right_UpperArm",
    "Right_Forearm",
    "Right_Hand",
    "Left_UpperLeg",
    "Left_LowerLeg",
    "Left_Foot",
    "Right_UpperLeg",
    "Right_LowerLeg",
    "Right_Foot",
}


class XsensSource:
    """Xsens MVN source backed by the optional parser SDK."""

    def __init__(
        self,
        *,
        port: int = 9763,
        fps: float = 60.0,
        human_height: float | None = None,
    ) -> None:
        """Configure the UDP source."""
        self._port = port
        self._fps = fps
        self._human_height = human_height
        self._device: _Device | None = None
        self._links: dict[str, str] = {}
        self._last_sample = -1
        self._initial_yaw_inverse: Rotation | None = None
        self._sequence = 0

    @property
    def source_format(self) -> str:
        """Profile source convention."""
        return "xsens_mvn"

    @property
    def fps(self) -> float:
        """Nominal source frame rate."""
        return self._fps

    def open(self) -> None:
        """Initialize and start the optional Xsens SDK."""
        try:
            module = import_module("xsens_mvn_robot")
        except ImportError as error:
            raise ValueError(
                "Xsens streaming requires the separately installed "
                "xsens_mvn_robot wheel"
            ) from error
        device: _Device = module.XsensWrapper(port=self._port)
        if not device.init():
            raise ConnectionError(
                f"failed to initialize Xsens UDP source on port {self._port}"
            )
        available = set(device.get_link_names())
        self._links = {
            target: source
            for source, target in _LINK_NAMES.items()
            if source in available
        }
        missing = sorted(_REQUIRED_BODIES - self._links.keys())
        if missing:
            names = ", ".join(missing)
            raise ConnectionError(f"Xsens stream is missing required bodies: {names}")
        device.start()
        self._device = device

    def read(self, timeout: float | None = None) -> HumanFrameSample | None:
        """Read the newest available Xsens frame."""
        if self._device is None:
            raise RuntimeError("Xsens source is not open")
        sample_counter = int(self._device.get_sample_counter())
        if sample_counter == self._last_sample:
            time.sleep(min(timeout or 0.001, 0.001))
            return None
        self._last_sample = sample_counter
        frame = {
            target: (
                np.asarray(self._device.get_link_position(source), dtype=np.float64),
                np.asarray(self._device.get_link_orientation(source), dtype=np.float64),
            )
            for target, source in self._links.items()
        }
        frame = self._normalize_initial_yaw(frame)
        sample = HumanFrameSample(
            sequence=self._sequence,
            frame=frame,
            received_monotonic=time.monotonic(),
            human_height=self._human_height,
        )
        self._sequence += 1
        return sample

    def close(self) -> None:
        """Stop the UDP source."""
        if self._device is None:
            return
        self._device.stop()
        self._device = None

    def _normalize_initial_yaw(self, frame: HumanFrame) -> HumanFrame:
        pelvis_rotation = Rotation.from_quat(frame["Pelvis"][1], scalar_first=True)
        if self._initial_yaw_inverse is None:
            yaw = pelvis_rotation.as_euler("ZYX")[0]
            self._initial_yaw_inverse = Rotation.from_euler("Z", -yaw)
        inverse = self._initial_yaw_inverse
        return {
            name: (
                inverse.apply(position),
                (inverse * Rotation.from_quat(rotation, scalar_first=True)).as_quat(
                    scalar_first=True
                ),
            )
            for name, (position, rotation) in frame.items()
        }
