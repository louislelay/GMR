"""Built-in and externally discoverable live-source factories."""

from dataclasses import dataclass
from importlib import metadata
from typing import Protocol

from ..streaming import LiveSource
from .pico import PicoSource
from .xsens import XsensSource


@dataclass(frozen=True)
class SourceOptions:
    """Common live-source command options."""

    fps: float | None = None
    port: int | None = None
    human_height: float | None = None


class SourceFactory(Protocol):
    """Factory loaded from the ``gmr.live_sources`` entry-point group."""

    def __call__(self, options: SourceOptions, /) -> LiveSource:
        """Create one unopened source."""
        ...


def create_live_source(name: str, options: SourceOptions) -> LiveSource:
    """Create a built-in or installed live source."""
    builtins: dict[str, SourceFactory] = {
        "pico": lambda value: PicoSource(fps=value.fps or 60.0),
        "xsens": lambda value: XsensSource(
            port=value.port or 9763,
            fps=value.fps or 60.0,
            human_height=value.human_height,
        ),
    }
    factories = dict(builtins)
    for entry_point in metadata.entry_points(group="gmr.live_sources"):
        if entry_point.name in factories:
            raise ValueError(f"duplicate live source: {entry_point.name!r}")
        factories[entry_point.name] = entry_point.load()
    try:
        return factories[name](options)
    except KeyError:
        available = ", ".join(sorted(factories))
        raise KeyError(
            f"unknown live source {name!r}; available: {available}"
        ) from None


__all__ = [
    "PicoSource",
    "SourceFactory",
    "SourceOptions",
    "XsensSource",
    "create_live_source",
]
