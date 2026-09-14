"""Regenerate the golden qpos trajectories used by the regression tests.

Run this only when an intentional behavior change is made to the retargeting
pipeline, and explain the change (and the golden diff) in the pull request:

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
