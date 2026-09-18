"""Composable robot and retargeting-profile catalog."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import mujoco as mj
import mujoco_menagerie

from .models import RetargetingProfile, RobotAssets, RobotSpec, TrackingCamera
from .profiles import load_profile


@dataclass(frozen=True)
class MenagerieAssets(RobotAssets):
    """Robot assets resolved through the pinned Menagerie cache."""

    robot_name: str
    model_entry: str
    scene_entry: str
    cache: mujoco_menagerie.Cache

    def load_model_spec(self) -> mj.MjSpec:
        """Load an editable robot-only specification."""
        return mujoco_menagerie.get(self.robot_name).spec(self.model_entry, self.cache)

    def load_scene_spec(self) -> mj.MjSpec:
        """Load an editable visualization-scene specification."""
        return mujoco_menagerie.get(self.robot_name).spec(self.scene_entry, self.cache)


@dataclass(frozen=True)
class ProfileRef:
    """Lazy profile reference contributed by a provider."""

    source_format: str
    robot: str
    path: Path

    def load(self) -> RetargetingProfile:
        """Load the referenced immutable profile."""
        return load_profile(
            self.path, source_format=self.source_format, robot=self.robot
        )


@dataclass(frozen=True)
class CatalogContribution:
    """Robots and profiles supplied atomically by one provider."""

    robots: tuple[RobotSpec, ...] = ()
    profiles: tuple[ProfileRef, ...] = ()


@dataclass(frozen=True)
class ProviderContext:
    """Injected resources available while providers build contributions."""

    menagerie_cache: mujoco_menagerie.Cache


class Provider(Protocol):
    """Callable extension provider registered under ``gmr.providers``."""

    def __call__(self, context: ProviderContext) -> CatalogContribution:
        """Build one catalog contribution."""
        ...


class Catalog:
    """Immutable lookup catalog for robots and retargeting profiles."""

    robots: Mapping[str, RobotSpec]
    profiles: Mapping[tuple[str, str], ProfileRef]

    def __init__(self, contributions: Iterable[CatalogContribution]) -> None:
        """Merge provider contributions and reject identifier collisions."""
        robots: dict[str, RobotSpec] = {}
        profiles: dict[tuple[str, str], ProfileRef] = {}
        for contribution in contributions:
            for robot in contribution.robots:
                if robot.identifier in robots:
                    raise ValueError(f"duplicate robot: {robot.identifier!r}")
                robots[robot.identifier] = robot
            for profile in contribution.profiles:
                key = (profile.source_format, profile.robot)
                if key in profiles:
                    raise ValueError(
                        f"duplicate profile for {profile.source_format!r} "
                        f"and {profile.robot!r}"
                    )
                profiles[key] = profile
        self.robots = MappingProxyType(robots)
        self.profiles = MappingProxyType(profiles)

    def robot(self, identifier: str) -> RobotSpec:
        """Return one robot or raise an actionable lookup error."""
        try:
            return self.robots[identifier]
        except KeyError:
            available = ", ".join(sorted(self.robots))
            raise KeyError(
                f"unknown robot {identifier!r}; available: {available}"
            ) from None

    def profile(self, source_format: str, robot: str) -> RetargetingProfile:
        """Load one source-to-robot profile."""
        key = (source_format, robot)
        try:
            return self.profiles[key].load()
        except KeyError:
            available = ", ".join(
                f"{source}->{target}" for source, target in sorted(self.profiles)
            )
            raise KeyError(
                f"no profile for {source_format!r} and {robot!r}; "
                f"available: {available}"
            ) from None


def _builtin_provider(context: ProviderContext) -> CatalogContribution:
    root = Path(__file__).parent / "profiles" / "builtin"

    def assets(name: str, model_entry: str, scene_entry: str) -> MenagerieAssets:
        return MenagerieAssets(
            robot_name=name,
            model_entry=model_entry,
            scene_entry=scene_entry,
            cache=context.menagerie_cache,
        )

    robots = (
        RobotSpec(
            "unitree_g1",
            assets("unitree_g1", "g1", "scene"),
            "pelvis",
            TrackingCamera("pelvis", 3.0),
        ),
        RobotSpec(
            "unitree_g1_with_hands",
            assets("unitree_g1", "g1_with_hands", "scene_with_hands"),
            "pelvis",
            TrackingCamera("pelvis", 3.0),
        ),
        RobotSpec(
            "unitree_h1",
            assets("unitree_h1", "h1", "scene"),
            "pelvis",
            TrackingCamera("pelvis", 3.0),
        ),
        RobotSpec(
            "booster_t1",
            assets("booster_t1", "t1", "scene"),
            "Trunk",
            TrackingCamera("Trunk", 3.0),
        ),
        RobotSpec(
            "fourier_n1",
            assets("fourier_n1", "n1", "scene"),
            "base_link",
            TrackingCamera("base_link", 3.0),
        ),
        RobotSpec(
            "pnd_adam_lite",
            assets("pndbotics_adam_lite", "adam_lite", "scene"),
            "pelvis",
            TrackingCamera("pelvis", 3.0),
        ),
        RobotSpec(
            "pal_talos",
            assets("pal_talos", "talos_position", "scene_position"),
            "base_link",
            TrackingCamera("base_link", 3.0),
        ),
    )
    paths = {
        ("smplx", "unitree_g1"): "smplx_to_g1.json",
        ("smplx", "unitree_g1_with_hands"): "smplx_to_g1.json",
        ("smplx", "unitree_h1"): "smplx_to_h1.json",
        ("smplx", "booster_t1"): "smplx_to_t1.json",
        ("smplx", "fourier_n1"): "smplx_to_n1.json",
        ("smplx", "pnd_adam_lite"): "smplx_to_adam.json",
        ("bvh_lafan1", "unitree_g1"): "bvh_lafan1_to_g1.json",
        ("bvh_lafan1", "fourier_n1"): "bvh_lafan1_to_n1.json",
        ("bvh_lafan1", "pal_talos"): "bvh_to_talos.json",
        ("bvh_nokov", "unitree_g1"): "bvh_nokov_to_g1.json",
        ("xsens_mvn", "unitree_g1"): "xsens_mvn_to_g1.json",
        ("xrobot", "unitree_g1"): "xrobot_to_g1.json",
    }
    profiles = tuple(
        ProfileRef(source, robot, root / filename)
        for (source, robot), filename in paths.items()
    )
    return CatalogContribution(robots=robots, profiles=profiles)


def build_catalog(
    providers: Iterable[Provider] = (),
    *,
    context: ProviderContext | None = None,
    discover: bool = True,
) -> Catalog:
    """Build a catalog from built-in, explicit, and installed providers."""
    resolved_context = context or ProviderContext(mujoco_menagerie.Cache())
    resolved_providers = list(providers)
    if discover:
        entry_points = metadata.entry_points(group="gmr.providers")
        resolved_providers.extend(entry_point.load() for entry_point in entry_points)
    contributions = [_builtin_provider(resolved_context)]
    contributions.extend(provider(resolved_context) for provider in resolved_providers)
    return Catalog(contributions)
