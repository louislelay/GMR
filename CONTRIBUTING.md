# Contributing to GMR

GMR welcomes focused fixes, retargeting profiles, robot support, and reusable
library improvements.

## Development setup

GMR supports Python 3.10.12 through 3.13 and uses
[uv](https://docs.astral.sh/uv/) for development:

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
uv sync
```

Before opening a pull request, run:

```bash
make check
```

Use `make format` to apply formatting and safe lint fixes, `make lint` to check
style, or `make test` to run the test suite independently.

Keep `uv.lock` synchronized when changing `pyproject.toml`.

## Change guidelines

- Keep changes small, typed, and covered by behavior-focused tests.
- Document public interfaces with concise docstrings.
- Pass robots, profiles, sources, and services explicitly; avoid mutable
  global registries.
- Keep robot-specific transforms in profiles rather than modifying vendor
  MJCF.
- Source built-in robot models from MuJoCo Menagerie.
- Preserve named-joint ordering. Update golden trajectories only when a
  retargeting change is intentional and explained.
- Prefer a clear API over compatibility shims for removed behavior.

Hardware-specific or optional integrations should generally use the extension
points described in the [extension guide](docs/extensions.md). See the
[profile guide](docs/ik_configs.md) when adding or changing a retargeting
profile.

## Pull requests

Explain the problem, the chosen approach, and any user-visible behavior change.
Include the validation commands you ran. Keep refactors separate from behavior
changes when practical so each commit remains easy to review.
