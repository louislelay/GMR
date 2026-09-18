# GMR: General Motion Retargeting

[![arXiv 2505.02833](https://img.shields.io/badge/arXiv-2505.02833-b31b1b.svg)](https://arxiv.org/abs/2505.02833)
[![arXiv 2510.02252](https://img.shields.io/badge/arXiv-2510.02252-b31b1b.svg)](https://arxiv.org/abs/2510.02252)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![GMR pipeline](https://raw.githubusercontent.com/YanjieZe/GMR/master/assets/GMR_pipeline.png)

GMR retargets offline and live human motion to humanoid robots. It provides a
typed Python API, canonical NPZ motion files, Menagerie-backed robot models,
and three focused commands:

- `retarget`: convert a source file or directory to robot motion.
- `visualize`: inspect, trim, and render motion in mjviser.
- `stream`: retarget live PICO, Xsens, or externally provided sources.

## Install

GMR requires Python 3.10.12–3.13.

```bash
uv add general_motion_retargeting
```

For repository development:

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
uv sync
make test
```

Robot models are downloaded and checksum-verified by the pinned
`mujoco-menagerie` package on first use. They are not duplicated in this
repository.

SMPL-X model files are licensed separately. Download them from the
[SMPL-X project](https://smpl-x.is.tue.mpg.de/) and place either `.npz` or
`.pkl` files under:

```text
<body-models>/smplx/SMPLX_NEUTRAL.npz
<body-models>/smplx/SMPLX_MALE.npz
<body-models>/smplx/SMPLX_FEMALE.npz
```

Set `GMR_SMPLX_MODELS=<body-models>` or pass `--body-models`.

## Commands

Retarget one motion:

```bash
uv run retarget \
  --robot unitree_g1 \
  --source smplx \
  --input walk.npz \
  --output walk_g1.npz
```

Directory input is recursive and its layout is mirrored under the output
directory:

```bash
uv run retarget \
  --robot unitree_g1 \
  --source bvh \
  --input motions/ \
  --output retargeted/
```

Open the browser workspace:

```bash
uv run visualize --robot-motion walk_g1.npz
uv run visualize \
  --robot-motion walk_g1.npz \
  --source-motion walk.npz \
  --source smplx
```

The workspace provides synchronized robot and source motion, timeline
scrubbing, playback speed, mocap and robot-site toggles, source and robot
skeletons, non-destructive time trimming, NPZ download, and viewport video
capture.

Retarget live Xsens or PICO data:

```bash
uv run stream --source xsens --robot unitree_g1 --port 9763
uv run stream --source pico --robot unitree_g1
```

GMR does not install platform-specific device SDKs. Follow the
[PICO setup guide](docs/pico.md) or [Xsens setup guide](docs/xsens.md) to
install and configure the required binding. Each binding is imported only when
its source is opened.

Installed commands can also be run through `uvx`:

```bash
uvx --from general_motion_retargeting retarget --help
uvx --from general_motion_retargeting visualize --help
uvx --from general_motion_retargeting stream --help
```

## Python API

Offline conversion uses the same application service as the command:

```python
from pathlib import Path

from general_motion_retargeting import RetargetApplication

application = RetargetApplication(body_models=Path("/models"))
motion = application.retarget_file(
    Path("walk.npz"),
    Path("walk_g1.npz"),
    source="smplx",
    robot="unitree_g1",
)
```

For direct frame control, inject catalog objects into `Retargeter`:

```python
from general_motion_retargeting import Retargeter, build_catalog

catalog = build_catalog()
retargeter = Retargeter(
    catalog.robot("unitree_g1"),
    catalog.profile("smplx", "unitree_g1"),
)
qpos = retargeter.retarget_frame(human_frame, human_height=1.8)
```

Live consumers subscribe independently. A slow viewer cannot consume or delay
frames intended for a policy:

```python
from general_motion_retargeting import RetargetedStream, SubscriptionMode

stream = RetargetedStream(source, retargeter)
policy = stream.subscribe(SubscriptionMode.LATEST)
recorder = stream.subscribe(SubscriptionMode.LOSSLESS)

with stream:
    frame = policy.wait_latest(after_sequence=-1, timeout=1.0)
```

Device timestamps and local monotonic timestamps remain separate.
`completed_monotonic - received_monotonic` measures local retargeting time;
it does not claim device-to-host latency without clock synchronization.

## Built-in robots

The initial validated Menagerie set is:

- `unitree_g1`
- `unitree_g1_with_hands`
- `unitree_h1`
- `booster_t1`
- `fourier_n1`
- `pnd_adam_lite`
- `pal_talos`

External packages can contribute robots and profiles through the
`gmr.providers` Python entry-point group, and live sources through
`gmr.live_sources`. See [the extension guide](docs/extensions.md).

Profiles use the named format documented in
[the profile guide](docs/ik_configs.md). Canonical robot motion uses validated,
pickle-free NPZ. Removed APIs and file formats remain available only from older
GMR revisions; see [the migration guide](docs/migration.md).

## Citation

```bibtex
@article{ze2025twist,
  title={TWIST: Teleoperated Whole-Body Imitation System},
  author={Ze, Yanjie and Li, Zixuan and Zeng, Yixuan and Zhang, Zhecheng
          and Liu, C. Karen and Xu, Huazhe},
  journal={arXiv preprint arXiv:2505.02833},
  year={2025}
}
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to propose changes. GMR is licensed
under the [MIT License](LICENSE).
