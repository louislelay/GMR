"""Tests that the limits passed to mink.solve_ik are actually enforced.

GeneralMotionRetargeting used to pass ik_limits positionally into
mink.solve_ik's ``safety_break`` parameter, so the list never reached the
solver: ``use_velocity_limit=True`` silently did nothing, and a truthy
``safety_break`` made mink raise (instead of warn) whenever the configuration
drifted marginally outside a joint limit.

Joint *position* limits were unaffected: mink applies a ConfigurationLimit by
default when ``limits`` is None, which is why the golden trajectories do not
change with the fix.
"""

import numpy as np
import pytest

import mink
from general_motion_retargeting import GeneralMotionRetargeting

from synthetic_motion import TPOSE_POSITIONS, TPOSE_QUAT

VELOCITY_LIMIT = 3.0 * np.pi


def build_frame(overrides):
    frame = {
        body: [np.array(position, dtype=float), np.array(TPOSE_QUAT, dtype=float)]
        for body, position in TPOSE_POSITIONS.items()
    }
    for body, position in overrides.items():
        frame[body][0] = np.array(position, dtype=float)
    return frame


def test_velocity_limit_enforced(monkeypatch):
    """With use_velocity_limit=True, no solve step may exceed 3*pi rad/s.

    The target teleports from a T-pose to arms-behind-the-back between two
    consecutive frames; without the velocity limit the solver commands joint
    velocities an order of magnitude above the cap (~100 rad/s).
    """
    recorded = []
    original = mink.solve_ik

    def recording_solve_ik(*args, **kwargs):
        velocity = original(*args, **kwargs)
        recorded.append(np.abs(velocity[6:]).max())
        return velocity

    monkeypatch.setattr(mink, "solve_ik", recording_solve_ik)

    retargeter = GeneralMotionRetargeting(
        "smplx", "unitree_g1", verbose=False, use_velocity_limit=True
    )
    tpose = build_frame({})
    arms_behind_back = build_frame(
        {
            "left_wrist": (-0.45, -0.10, 0.55),
            "right_wrist": (-0.45, 0.10, 0.55),
            "left_elbow": (-0.25, 0.05, 0.90),
            "right_elbow": (-0.25, -0.05, 0.90),
        }
    )
    retargeter.retarget(tpose)
    retargeter.retarget(arms_behind_back)

    assert recorded, "solve_ik was never called"
    worst = float(max(recorded))
    assert worst <= VELOCITY_LIMIT + 1e-6, (
        f"joint velocity {worst:.2f} rad/s exceeds the "
        f"{VELOCITY_LIMIT:.2f} rad/s limit"
    )


def test_safety_break_not_engaged():
    """solve_ik must receive safety_break=False (the mink default).

    When ik_limits was passed positionally it landed in safety_break, making
    mink raise NotWithinConfigurationLimits whenever numerical drift pushed a
    joint marginally outside its range mid-motion.
    """
    retargeter = GeneralMotionRetargeting("smplx", "unitree_g1", verbose=False)
    seen = []
    original = mink.solve_ik

    def inspecting_solve_ik(*args, **kwargs):
        # args: configuration, tasks, dt, solver, damping, [safety_break, ...]
        safety_break = args[5] if len(args) > 5 else kwargs.get("safety_break", False)
        seen.append(bool(safety_break))
        return original(*args, **kwargs)

    mink.solve_ik = inspecting_solve_ik
    try:
        retargeter.retarget(build_frame({}))
    finally:
        mink.solve_ik = original
    assert seen and not any(seen)
