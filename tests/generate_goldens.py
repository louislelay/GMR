"""Regenerate the golden qpos trajectories used by the regression tests.

Run this only when an intentional behavior change is made to the retargeting
pipeline. Explain the change in the pull request and attach a rendered video
of the new goldens so the motion can be certified by eye:

    python tests/generate_goldens.py
"""

import pathlib

import numpy as np

from test_retarget_regression import ROBOTS, load_synthetic_motion, retarget_motion

DATA_DIR = pathlib.Path(__file__).parent / "data"


def main():
    frames = load_synthetic_motion()
    for robot in ROBOTS:
        qpos = retarget_motion(robot, frames)
        out = DATA_DIR / f"golden_qpos_{robot}.npy"
        np.save(out, qpos)
        print(f"wrote {qpos.shape} qpos to {out}")


if __name__ == "__main__":
    main()
