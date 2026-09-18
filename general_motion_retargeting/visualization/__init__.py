"""Offline and live mjviser workspaces."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live import LiveWorkspace as LiveWorkspace
    from .motion import MotionWorkspace as MotionWorkspace

_EXPORTS = {
    "LiveWorkspace": (".live", "LiveWorkspace"),
    "MotionWorkspace": (".motion", "MotionWorkspace"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
