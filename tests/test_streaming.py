"""Tests for reusable multi-consumer streaming semantics."""

import sys
import threading
import time
from types import ModuleType

import numpy as np
import pytest

from general_motion_retargeting.models import HumanFrame
from general_motion_retargeting.sources.xsens import XsensSource
from general_motion_retargeting.streaming import (
    FramePublisher,
    HumanFrameSample,
    RetargetedFrame,
    RetargetedStream,
    SubscriptionMode,
)


class _Profile:
    source_format = "test"


class _Robot:
    identifier = "test_robot"


class _Retargeter:
    profile = _Profile()
    robot = _Robot()

    def __init__(self) -> None:
        self.calls = 0

    def retarget_frame(
        self, frame: HumanFrame, *, human_height: float | None = None
    ) -> np.ndarray:
        del frame, human_height
        self.calls += 1
        return np.arange(8, dtype=np.float64)


class _Source:
    source_format = "test"
    fps = 60.0

    def __init__(self, sample: HumanFrameSample) -> None:
        self._sample: HumanFrameSample | None = sample
        self._opened = False
        self._closed = threading.Event()

    def open(self) -> None:
        self._opened = True

    def read(self, timeout: float | None = None) -> HumanFrameSample | None:
        if not self._opened:
            raise RuntimeError("source is not open")
        if self._sample is not None:
            sample = self._sample
            self._sample = None
            return sample
        self._closed.wait(timeout)
        return None

    def close(self) -> None:
        self._closed.set()


def _sample() -> HumanFrameSample:
    return HumanFrameSample(
        sequence=42,
        frame={
            "root": (
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        },
        received_monotonic=time.monotonic(),
        source_timestamp=123.0,
    )


def _retargeted(sequence: int) -> RetargetedFrame:
    now = time.monotonic()
    return RetargetedFrame(
        sequence=sequence,
        source_sequence=sequence,
        source_timestamp=None,
        received_monotonic=now,
        completed_monotonic=now,
        robot="test",
        qpos=np.zeros(8),
        source_frame=_sample().frame,
    )


def test_each_consumer_receives_same_retargeted_frame_once() -> None:
    source = _Source(_sample())
    retargeter = _Retargeter()
    stream = RetargetedStream(source, retargeter)
    policy = stream.subscribe()
    viewer = stream.subscribe()

    with stream:
        policy_frame = policy.wait_latest(-1, timeout=1.0)
        viewer_frame = viewer.wait_latest(-1, timeout=1.0)

    assert policy_frame is viewer_frame
    assert policy_frame.source_sequence == 42
    assert retargeter.calls == 1
    with pytest.raises(ValueError):
        policy_frame.qpos[0] = 10.0


def test_latest_subscription_reports_overwrites() -> None:
    publisher = FramePublisher()
    latest = publisher.subscribe(SubscriptionMode.LATEST)

    publisher.publish(_retargeted(0))
    publisher.publish(_retargeted(1))
    frame = latest.read_latest()

    assert frame is not None
    assert frame.sequence == 1
    assert latest.dropped_count == 1


def test_lossless_subscription_can_use_bounded_queue() -> None:
    publisher = FramePublisher()
    recorder = publisher.subscribe(SubscriptionMode.LOSSLESS, capacity=1)
    publisher.publish(_retargeted(0))
    publisher.publish(_retargeted(1))

    frame = recorder.read()
    assert frame is not None
    assert frame.sequence == 0
    assert recorder.dropped_count == 1


def test_xsens_source_loads_optional_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = False

    class Device:
        def __init__(self, *, port: int) -> None:
            assert port == 1234

        def init(self) -> bool:
            return True

        def start(self) -> None:
            return None

        def stop(self) -> None:
            nonlocal stopped
            stopped = True

        def get_sample_counter(self) -> int:
            return 1

        def get_link_names(self) -> list[str]:
            return [
                "pelvis",
                "t8",
                "left_upper_arm",
                "left_forearm",
                "left_hand",
                "right_upper_arm",
                "right_forearm",
                "right_hand",
                "left_upper_leg",
                "left_lower_leg",
                "left_foot",
                "right_upper_leg",
                "right_lower_leg",
                "right_foot",
            ]

        def get_link_position(self, name: str) -> np.ndarray:
            del name
            return np.zeros(3)

        def get_link_orientation(self, name: str) -> np.ndarray:
            del name
            return np.array([1.0, 0.0, 0.0, 0.0])

    module = ModuleType("xsens_mvn_robot")
    module.XsensWrapper = Device
    monkeypatch.setitem(sys.modules, module.__name__, module)

    source = XsensSource(port=1234)
    source.open()
    sample = source.read()
    source.close()

    assert sample is not None
    assert set(sample.frame) >= {"Pelvis", "Left_Hand", "Right_Foot"}
    assert stopped


def test_xsens_source_reports_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "xsens_mvn_robot", raising=False)

    def missing_sdk(name: str) -> ModuleType:
        raise ImportError(name)

    monkeypatch.setattr(
        "general_motion_retargeting.sources.xsens.import_module",
        missing_sdk,
    )

    with pytest.raises(ValueError, match="xsens_mvn_robot"):
        XsensSource().open()
