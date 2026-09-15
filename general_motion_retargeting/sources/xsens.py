"""Xsens MVN UDP live source."""

import time
from typing import Protocol

from ..models import HumanFrame
from ..streaming import HumanFrameSample


class _Adapter(Protocol):
    def initialize(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_human_frame(self) -> HumanFrame | None: ...


class XsensSource:
    """Xsens MVN source backed by GMR's UDP adapter."""

    def __init__(
        self,
        *,
        port: int = 9763,
        fps: float = 60.0,
        human_height: float | None = None,
        verbose: bool = True,
    ) -> None:
        """Configure the UDP source.

        Args:
            port: MVN UDP port.
            fps: Nominal source frame rate.
            human_height: Optional measured performer height.
            verbose: Enable adapter status messages.
        """
        self._port = port
        self._fps = fps
        self._human_height = human_height
        self._verbose = verbose
        self._adapter: _Adapter | None = None
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
        """Initialize and start the UDP adapter."""
        try:
            from ..utils.xsens_vendor.xsens_to_gmr_adapter import XsensToGMR
        except ImportError as error:
            raise ValueError(
                "Xsens streaming requires an installed xsens_mvn_robot adapter"
            ) from error

        adapter = XsensToGMR(port=self._port, verbose=self._verbose)
        if not adapter.initialize():
            raise ConnectionError(
                f"failed to initialize Xsens UDP source on port {self._port}"
            )
        adapter.start()
        self._adapter = adapter

    def read(self, timeout: float | None = None) -> HumanFrameSample | None:
        """Read the newest available Xsens frame."""
        if self._adapter is None:
            raise RuntimeError("Xsens source is not open")
        frame = self._adapter.get_human_frame()
        if frame is None:
            time.sleep(min(timeout or 0.001, 0.001))
            return None
        sample = HumanFrameSample(
            sequence=self._sequence,
            frame=frame,
            received_monotonic=time.monotonic(),
            human_height=self._human_height,
        )
        self._sequence += 1
        return sample

    def close(self) -> None:
        """Stop the UDP adapter."""
        if self._adapter is None:
            return
        self._adapter.stop()
        self._adapter = None
