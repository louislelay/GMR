"""Generate the deterministic synthetic human motion used by the regression tests.

The motion is a simple parametric "march in place while moving forward" for a
1.8 m human, expressed in the SMPL-X global-frame convention used by GMR:
z-up world, subject facing +x, one (position, wxyz quaternion) pair per body.

All quaternions are the constant T-pose global rotation of an upright z-up
person facing +x, which in SMPL-X convention (y-up, facing +z, left = +x) is
the 120-degree rotation about (1, 1, 1): (0.5, 0.5, 0.5, 0.5).

The file is committed at tests/data/synthetic_motion.json. Regenerate with:

    python tests/generate_synthetic_motion.py
"""

import json
import math
import pathlib

FPS = 30
NUM_FRAMES = 90
STRIDE_HZ = 1.0
FORWARD_SPEED = 0.3

TPOSE_QUAT = [0.5, 0.5, 0.5, 0.5]

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


def build_frame(t):
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
        frame[body] = [round(v, 6) for v in (px, py, pz)] + TPOSE_QUAT
    return frame


def main():
    frames = [build_frame(i / FPS) for i in range(NUM_FRAMES)]
    out = pathlib.Path(__file__).parent / "data" / "synthetic_motion.json"
    # One frame per line so the file stays diffable and skimmable in review.
    frame_lines = ",\n".join(
        json.dumps(frame, separators=(",", ":")) for frame in frames
    )
    out.write_text(f'{{"fps":{FPS},"frames":[\n{frame_lines}\n]}}\n')
    print(f"wrote {NUM_FRAMES} frames to {out}")


if __name__ == "__main__":
    main()
