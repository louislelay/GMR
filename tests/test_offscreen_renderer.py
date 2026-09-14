"""Smoke test for headless video rendering."""

import mujoco as mj
import numpy as np
import pytest
from test_retarget_regression import DATA_DIR

from general_motion_retargeting.offscreen_renderer import render_robot_motion


def _gl_available() -> bool:
    """Probe whether an offscreen GL context can be created at all."""
    try:
        model = mj.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        renderer = mj.Renderer(model, height=32, width=32)
        renderer.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gl_available(), reason="no offscreen GL backend available"
)


def test_render_golden_frames_to_mp4(tmp_path):
    qpos = np.load(DATA_DIR / "golden_qpos_unitree_g1.npy")[:5]
    out = tmp_path / "golden.mp4"
    render_robot_motion("unitree_g1", qpos[:, :3], qpos[:, 3:7], qpos[:, 7:], 30, out)
    assert out.exists()
    assert out.stat().st_size > 0
