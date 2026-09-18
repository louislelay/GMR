# Migration guide

This release is a clean break. It does not retain compatibility facades,
deprecated scripts, positional profiles, pickle readers, or global parameter
dictionaries. Pin an older GMR commit if an application still requires those
interfaces.

## Commands

Use the three installed commands directly:

```text
retarget --source smplx --robot unitree_g1 --input motion.npz --output robot.npz
visualize --robot-motion robot.npz
stream --source pico --robot unitree_g1
```

Video is no longer a retarget or playback flag. Open `visualize`, choose a
non-destructive time range, then render and download the current browser
viewport.

## Robot identifiers and assets

Built-in robot files now come from the pinned, checksum-verified Menagerie
cache. The retained identifiers are listed in the README.

Registrations without directly validated Menagerie compatibility were
removed. Publish those robots and their profiles from an external provider
package; do not restore copied assets to GMR. Similarly named Menagerie models
must be treated as new targets until their frames, joints, and golden
trajectories are validated.

Resolve robots and profiles through the catalog:

```python
from general_motion_retargeting import build_catalog

catalog = build_catalog()
robot = catalog.robot("unitree_g1")
profile = catalog.profile("smplx", "unitree_g1")
```

## Retargeter

Inject `RobotSpec`, `RetargetingProfile`, and optional `SolverSettings`
directly into `Retargeter`.

Profiles use named human and robot frames. GMR creates collision-free MuJoCo
sites from robot-side transforms through `MjSpec`; vendor MJCF files no longer
need hand-authored tracking sites.

## Motion files

Canonical `RobotMotion` NPZ stores:

- schema version
- robot, profile, and source identifiers
- frame rate and time origin
- ordered scalar joint names
- root positions and wxyz quaternions
- joint positions

Loading uses `allow_pickle=False`. There is no pickle fallback or migration
API. Convert trusted old files with the GMR revision that created them, then
upgrade the resulting data to canonical NPZ at the application boundary.

## Live consumers

Do not share one iterator between a viewer, policy, and recorder. Create one
subscription per consumer:

- `LATEST` replaces stale frames and reports its drop count.
- `LOSSLESS` uses an unbounded queue by default.
- A bounded `LOSSLESS` queue retains old frames and reports rejected new
  frames through `dropped_count`.

`wait_latest(after_sequence, timeout)` blocks without busy polling. Source
device time, host receive time, and retarget completion time remain separate.
