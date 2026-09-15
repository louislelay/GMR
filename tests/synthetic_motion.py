"""Deterministic synthetic human motion used by the regression tests.

The motion is a simple parametric "march in place while moving forward" for a
1.8 m human, expressed in the SMPL-X global-frame convention used by GMR:
z-up world, subject facing +x, one (position, wxyz quaternion) pair per body.

All quaternions are the constant T-pose global rotation of an upright z-up
person facing +x, which in SMPL-X convention (y-up, facing +z, left = +x) is
the 120-degree rotation about (1, 1, 1): (0.5, 0.5, 0.5, 0.5).

Frames are generated in memory, in the exact input format of
GeneralMotionRetargeting.retarget(). The goldens in tests/data were produced
from these frames (see test_retarget_regression.py for how to regenerate
them); positions are rounded to 1e-6 m so the frames are bit-stable across
platforms.
"""

import math

import numpy as np

FPS = 30
NUM_FRAMES = 90
STRIDE_HZ = 1.0
FORWARD_SPEED = 0.3

TPOSE_QUAT = (0.5, 0.5, 0.5, 0.5)

# T-pose body positions for a 1.8 m human, z-up, facing +x, left = +y.
TPOSE_POSITIONS = {
    "pelvis": (0.0, 0.0, 0.95),
    "spine3": (0.0, 0.0, 1.25),
    "left_hip": (0.0, 0.09, 0.90),
    "right_hip": (0.0, -0.09, 0.90),
    "left_knee": (0.0, 0.10, 0.50),
    "right_knee": (0.0, -0.10, 0.50),
    "left_foot": (0.0, 0.11, 0.08),
    "right_foot": (0.0, -0.11, 0.08),
    "left_shoulder": (0.0, 0.18, 1.40),
    "right_shoulder": (0.0, -0.18, 1.40),
    "left_elbow": (0.0, 0.45, 1.40),
    "right_elbow": (0.0, -0.45, 1.40),
    "left_wrist": (0.0, 0.70, 1.40),
    "right_wrist": (0.0, -0.70, 1.40),
}

Frame = dict[str, tuple[np.ndarray, np.ndarray]]


def _build_frame(t: float) -> Frame:
    phase = 2.0 * math.pi * STRIDE_HZ * t
    swing = math.sin(phase)
    forward = FORWARD_SPEED * t
    bob = 0.03 * math.sin(2.0 * phase)

    frame = {}
    for body, (x, y, z) in TPOSE_POSITIONS.items():
        px, py, pz = x + forward, y, z + bob
        if body == "left_foot":
            pz += 0.06 * max(0.0, swing)
            px += 0.08 * swing
        if body == "right_foot":
            pz += 0.06 * max(0.0, -swing)
            px -= 0.08 * swing
        if body in ("left_elbow", "left_wrist"):
            px -= 0.10 * swing
        if body in ("right_elbow", "right_wrist"):
            px += 0.10 * swing
        pos = np.array([round(v, 6) for v in (px, py, pz)])
        frame[body] = (pos, np.array(TPOSE_QUAT))
    return frame


def build_frames() -> list[Frame]:
    """All frames of the synthetic motion, ready to feed to retarget()."""
    return [_build_frame(i / FPS) for i in range(NUM_FRAMES)]
