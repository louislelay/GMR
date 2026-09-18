"""Public GMR API."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .application import RetargetApplication
    from .assets import PathRobotAssets
    from .catalog import (
        Catalog,
        CatalogContribution,
        MenagerieAssets,
        ProfileRef,
        Provider,
        ProviderContext,
        build_catalog,
    )
    from .models import (
        FrameTransform,
        HumanFrame,
        HumanMotion,
        Match,
        RetargetingProfile,
        RobotMotion,
        RobotSpec,
        SolverSettings,
        TrackingCamera,
    )
    from .motion_io import SCHEMA_VERSION, load_robot_motion, save_robot_motion
    from .profiles import load_profile, validate_human_frame
    from .retargeter import Retargeter
    from .streaming import (
        FramePublisher,
        FrameSubscription,
        HumanFrameSample,
        LiveSource,
        RetargetedFrame,
        RetargetedStream,
        SubscriptionMode,
    )
    from .visualization import LiveWorkspace, MotionWorkspace

_EXPORTS = {
    "SCHEMA_VERSION": (".motion_io", "SCHEMA_VERSION"),
    "Catalog": (".catalog", "Catalog"),
    "CatalogContribution": (".catalog", "CatalogContribution"),
    "FramePublisher": (".streaming", "FramePublisher"),
    "FrameSubscription": (".streaming", "FrameSubscription"),
    "FrameTransform": (".models", "FrameTransform"),
    "HumanFrame": (".models", "HumanFrame"),
    "HumanFrameSample": (".streaming", "HumanFrameSample"),
    "HumanMotion": (".models", "HumanMotion"),
    "LiveSource": (".streaming", "LiveSource"),
    "LiveWorkspace": (".visualization", "LiveWorkspace"),
    "Match": (".models", "Match"),
    "MenagerieAssets": (".catalog", "MenagerieAssets"),
    "MotionWorkspace": (".visualization", "MotionWorkspace"),
    "PathRobotAssets": (".assets", "PathRobotAssets"),
    "ProfileRef": (".catalog", "ProfileRef"),
    "Provider": (".catalog", "Provider"),
    "ProviderContext": (".catalog", "ProviderContext"),
    "RetargetApplication": (".application", "RetargetApplication"),
    "RetargetedFrame": (".streaming", "RetargetedFrame"),
    "RetargetedStream": (".streaming", "RetargetedStream"),
    "Retargeter": (".retargeter", "Retargeter"),
    "RetargetingProfile": (".models", "RetargetingProfile"),
    "RobotMotion": (".models", "RobotMotion"),
    "RobotSpec": (".models", "RobotSpec"),
    "SolverSettings": (".models", "SolverSettings"),
    "SubscriptionMode": (".streaming", "SubscriptionMode"),
    "TrackingCamera": (".models", "TrackingCamera"),
    "build_catalog": (".catalog", "build_catalog"),
    "load_profile": (".profiles", "load_profile"),
    "load_robot_motion": (".motion_io", "load_robot_motion"),
    "save_robot_motion": (".motion_io", "save_robot_motion"),
    "validate_human_frame": (".profiles", "validate_human_frame"),
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
