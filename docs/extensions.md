# Extending GMR

External packages can add robots, profiles, and live sources without importing
their application code into GMR.

## Robot and profile provider

Implement a callable receiving `ProviderContext` and returning one
`CatalogContribution`:

```python
from pathlib import Path

from general_motion_retargeting import (
    CatalogContribution,
    PathRobotAssets,
    ProfileRef,
    ProviderContext,
    RobotSpec,
    TrackingCamera,
)


def provide_astro(context: ProviderContext) -> CatalogContribution:
    del context
    package_root = Path(__file__).parent
    return CatalogContribution(
        robots=(
            RobotSpec(
                identifier="uma_astro_biped",
                assets=PathRobotAssets(
                    package_root / "assets" / "astro.xml",
                    package_root / "assets" / "scene.xml",
                ),
                root_body="base",
                camera=TrackingCamera(body="base", distance=3.0),
            ),
        ),
        profiles=(
            ProfileRef(
                source_format="smplx",
                robot="uma_astro_biped",
                path=package_root / "profiles" / "smplx_to_astro.json",
            ),
        ),
    )
```

Register it in the extension package:

```toml
[project.entry-points."gmr.providers"]
astro = "nexus_gmr:provide_astro"
```

Provider identifiers must be unique. GMR rejects collisions instead of
silently replacing a built-in definition. `ProviderContext` carries shared
resources such as the configured Menagerie cache; providers should use the
context instead of creating process-global clients.

`PathRobotAssets` is appropriate for package-owned MJCF. `MenagerieAssets` can
refer to a model from the pinned Menagerie registry. Both return a fresh
`MjSpec`, allowing GMR to inject profile tracking sites without changing
vendor files.

## Live source

A live source implements `LiveSource`: `source_format`, `fps`, `open`,
`read(timeout)`, and `close`. `read` returns timestamped `HumanFrameSample`
objects.

Register a factory taking `SourceOptions`:

```toml
[project.entry-points."gmr.live_sources"]
my_tracker = "tracker_gmr:create_source"
```

```python
from general_motion_retargeting.sources import SourceOptions


def create_source(options: SourceOptions) -> MyTrackerSource:
    return MyTrackerSource(
        fps=options.fps or 60.0,
        port=options.port or 9000,
    )
```

Import native SDKs inside `open`, not at module import. This keeps package
discovery and non-streaming workflows usable on unsupported platforms.

## Downstream integration

Downstream projects should contribute robot-specific definitions and consume
canonical `RobotMotion` or `RetargetedFrame`. They should not mutate GMR
registries, duplicate the retarget loop, or make GMR import the downstream
project.

For an mjlab-style training pipeline, keep the boundary:

```text
source motion -> GMR RobotMotion -> downstream named-joint importer
```

Simulation-specific body-state recording and training dataset conversion
belong in the downstream adapter.
