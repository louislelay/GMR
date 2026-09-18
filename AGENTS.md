# GMR contributor guide

GMR is a typed Python library and command-line application for retargeting
human motion to humanoid robots.

## Development

```bash
uv sync
uv run ruff format --check general_motion_retargeting tests
uv run ruff check general_motion_retargeting tests
uv run pytest
```

Python 3.10.12 through 3.13 is supported. Keep `uv.lock` synchronized with
`pyproject.toml`.

## Architecture

- `Retargeter` owns one stateful MuJoCo/Mink differential-IK session.
- `Catalog` resolves built-in Menagerie robots, profiles, and external
  providers.
- `RobotMotion` is the canonical immutable trajectory; `motion_io` persists it
  as pickle-free NPZ.
- `RetargetApplication` is shared by the `retarget` and `visualize` commands.
- `RetargetedStream` fans live results out to independent consumers.
- `MotionWorkspace` and `LiveWorkspace` are the mjviser frontends.

Built-in MJCF must come from MuJoCo Menagerie. External packages can register
robots and profiles through `gmr.providers`, and live sources through
`gmr.live_sources`.

## Change guidelines

- Prefer a clean API over compatibility shims; old behavior remains available
  in older revisions.
- Pass robots, profiles, sources, and services explicitly. Do not add mutable
  global registries.
- Keep public interfaces typed and documented with short docstrings.
- Keep robot-specific transforms in versioned profiles, not vendor MJCF.
- Add tests for behavior and contracts, not implementation details.
- Preserve named-joint ordering and update golden trajectories only for
  intentional retargeting changes.
