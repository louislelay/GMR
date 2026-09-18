"""Regression test for forwarding limits to Mink."""

import mink
import numpy as np
import pytest
from synthetic_motion import build_frames
from test_retarget_regression import build_retargeter

from general_motion_retargeting import SolverSettings


def test_solve_ik_receives_limits_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass IK limits by keyword without setting safety_break."""
    retargeter = build_retargeter(
        "unitree_g1", SolverSettings(use_velocity_limits=True)
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def recording_solve_ik(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros(retargeter.model.nv)

    monkeypatch.setattr(mink, "solve_ik", recording_solve_ik)
    retargeter.retarget_frame(build_frames()[0])

    assert calls
    limits = calls[0][1]["limits"]
    for args, kwargs in calls:
        assert len(args) == 5
        assert kwargs["limits"] is limits
        assert "safety_break" not in kwargs
