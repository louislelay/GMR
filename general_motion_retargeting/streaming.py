"""Reusable live retargeting with independent consumer subscriptions."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .models import FloatArray, HumanFrame, _readonly_frame

if TYPE_CHECKING:
    from .retargeter import Retargeter


@dataclass(frozen=True)
class HumanFrameSample:
    """One timestamped sample produced by a live source."""

    sequence: int
    frame: HumanFrame
    received_monotonic: float
    source_timestamp: float | None = None
    human_height: float | None = None

    def __post_init__(self) -> None:
        """Own and freeze the frame arrays."""
        object.__setattr__(self, "frame", _readonly_frame(self.frame))


@dataclass(frozen=True)
class RetargetedFrame:
    """One immutable retargeted frame published to every consumer."""

    sequence: int
    source_sequence: int
    source_timestamp: float | None
    received_monotonic: float
    completed_monotonic: float
    robot: str
    qpos: FloatArray
    source_frame: HumanFrame

    def __post_init__(self) -> None:
        """Own and freeze qpos and visualization source data."""
        qpos = np.asarray(self.qpos, dtype=np.float64).copy()
        qpos.setflags(write=False)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "source_frame", _readonly_frame(self.source_frame))


class LiveSource(Protocol):
    """Timestamped human-frame source owned by one retargeting worker."""

    @property
    def source_format(self) -> str:
        """Profile source format emitted by this adapter."""
        ...

    @property
    def fps(self) -> float:
        """Nominal source frame rate."""
        ...

    def open(self) -> None:
        """Acquire source resources."""
        ...

    def read(self, timeout: float | None = None) -> HumanFrameSample | None:
        """Read the next source frame, or none on timeout."""
        ...

    def close(self) -> None:
        """Release source resources."""
        ...


class SubscriptionMode(Enum):
    """Consumer queue behavior."""

    LATEST = "latest"
    LOSSLESS = "lossless"


class FrameSubscription:
    """One consumer-owned queue or latest-value slot."""

    def __init__(
        self,
        mode: SubscriptionMode,
        capacity: int | None,
    ) -> None:
        """Create an unpublished subscription."""
        if mode is SubscriptionMode.LATEST:
            capacity = 1
        if capacity is not None and capacity <= 0:
            raise ValueError("subscription capacity must be positive")
        self.mode = mode
        self.capacity = capacity
        self._frames: deque[RetargetedFrame] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._dropped_count = 0

    @property
    def dropped_count(self) -> int:
        """Number of frames discarded by this subscription."""
        with self._condition:
            return self._dropped_count

    def _publish(self, frame: RetargetedFrame) -> None:
        with self._condition:
            if self._closed:
                return
            if self.mode is SubscriptionMode.LATEST and self._frames:
                self._dropped_count += len(self._frames)
                self._frames.clear()
            if self.capacity is not None and len(self._frames) >= self.capacity:
                self._dropped_count += 1
                return
            self._frames.append(frame)
            self._condition.notify_all()

    def read(self, timeout: float | None = None) -> RetargetedFrame | None:
        """Read the oldest queued frame, waiting up to timeout seconds."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: bool(self._frames) or self._closed,
                timeout=timeout,
            )
            if not ready or not self._frames:
                return None
            return self._frames.popleft()

    def read_latest(self) -> RetargetedFrame | None:
        """Return the newest queued frame without blocking."""
        with self._condition:
            if not self._frames:
                return None
            frame = self._frames[-1]
            self._frames.clear()
            return frame

    def wait_latest(
        self, after_sequence: int, timeout: float | None = None
    ) -> RetargetedFrame | None:
        """Wait for and consume the newest frame after a sequence number."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    (bool(self._frames) and self._frames[-1].sequence > after_sequence)
                    or self._closed
                ),
                timeout=timeout,
            )
            if not ready or not self._frames:
                return None
            frame = self._frames[-1]
            if frame.sequence <= after_sequence:
                return None
            self._frames.clear()
            return frame

    def close(self) -> None:
        """Close this subscription and wake blocked readers."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FramePublisher:
    """Fan one immutable frame out to independent subscriptions."""

    def __init__(self) -> None:
        """Create an empty publisher."""
        self._subscriptions: set[FrameSubscription] = set()
        self._lock = threading.Lock()
        self._closed = False

    def subscribe(
        self,
        mode: SubscriptionMode = SubscriptionMode.LATEST,
        *,
        capacity: int | None = None,
    ) -> FrameSubscription:
        """Create and register a consumer subscription."""
        subscription = FrameSubscription(mode, capacity)
        with self._lock:
            if self._closed:
                subscription.close()
                return subscription
            self._subscriptions.add(subscription)
        return subscription

    def publish(self, frame: RetargetedFrame) -> None:
        """Publish one frame to every current subscriber."""
        with self._lock:
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            subscription._publish(frame)

    def close(self) -> None:
        """Close the publisher and every subscription."""
        with self._lock:
            self._closed = True
            subscriptions = tuple(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.close()


class RetargetedStream:
    """Own one source and retargeter worker, publishing each result once."""

    def __init__(
        self,
        source: LiveSource,
        retargeter: Retargeter,
        *,
        publisher: FramePublisher | None = None,
    ) -> None:
        """Prepare a stream without starting its worker."""
        if source.source_format != retargeter.profile.source_format:
            raise ValueError(
                f"source emits {source.source_format!r}, profile expects "
                f"{retargeter.profile.source_format!r}"
            )
        self.source = source
        self.retargeter = retargeter
        self.publisher = publisher or FramePublisher()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        """Worker error, when the source or retargeter failed."""
        return self._error

    def subscribe(
        self,
        mode: SubscriptionMode = SubscriptionMode.LATEST,
        *,
        capacity: int | None = None,
    ) -> FrameSubscription:
        """Create an independent consumer subscription."""
        return self.publisher.subscribe(mode, capacity=capacity)

    def start(self) -> None:
        """Open the source and start the retargeting worker."""
        if self._thread is not None:
            raise RuntimeError("stream has already been started")
        self.source.open()
        self._thread = threading.Thread(
            target=self._run,
            name="gmr-retargeted-stream",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        sequence = 0
        try:
            while not self._stop.is_set():
                sample = self.source.read(timeout=0.1)
                if sample is None:
                    continue
                qpos = self.retargeter.retarget_frame(
                    sample.frame, human_height=sample.human_height
                )
                completed = time.monotonic()
                self.publisher.publish(
                    RetargetedFrame(
                        sequence=sequence,
                        source_sequence=sample.sequence,
                        source_timestamp=sample.source_timestamp,
                        received_monotonic=sample.received_monotonic,
                        completed_monotonic=completed,
                        robot=self.retargeter.robot.identifier,
                        qpos=qpos,
                        source_frame=sample.frame,
                    )
                )
                sequence += 1
        except BaseException as error:
            self._error = error
        finally:
            self.publisher.close()

    def close(self) -> None:
        """Stop the worker and release the source."""
        self._stop.set()
        self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise TimeoutError("live source did not stop within two seconds")
        self.publisher.close()

    def __enter__(self) -> RetargetedStream:
        """Start and return this stream."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close this stream."""
        del exc_type, exc_value, traceback
        self.close()
