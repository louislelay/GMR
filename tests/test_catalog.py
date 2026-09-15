"""Tests for built-in and external catalog providers."""

from pathlib import Path

import mujoco_menagerie
import pytest

from general_motion_retargeting import (
    RobotSpec,
    TrackingCamera,
)
from general_motion_retargeting.assets import PathRobotAssets
from general_motion_retargeting.catalog import (
    CatalogContribution,
    ProviderContext,
    build_catalog,
)
from general_motion_retargeting.retargeter import Retargeter


def test_all_builtin_profiles_compile_with_menagerie() -> None:
    catalog = build_catalog(discover=False)

    for source_format, robot in catalog.profiles:
        Retargeter(
            catalog.robot(robot),
            catalog.profile(source_format, robot),
        )


def test_explicit_provider_contributes_robot() -> None:
    def provider(context: ProviderContext) -> CatalogContribution:
        del context
        return CatalogContribution(
            robots=(
                RobotSpec(
                    identifier="external",
                    assets=PathRobotAssets(Path("external.xml")),
                    root_body="base",
                    camera=TrackingCamera("base", 2.0),
                ),
            )
        )

    extended = build_catalog(
        (provider,),
        context=ProviderContext(mujoco_menagerie.Cache()),
        discover=False,
    )

    assert extended.robot("external").identifier == "external"


def test_provider_collision_is_rejected() -> None:
    context = ProviderContext(mujoco_menagerie.Cache())
    robot = build_catalog(context=context, discover=False).robot("unitree_g1")

    def provider(_: ProviderContext) -> CatalogContribution:
        return CatalogContribution(robots=(robot,))

    with pytest.raises(ValueError, match="duplicate robot"):
        build_catalog((provider,), context=context, discover=False)
