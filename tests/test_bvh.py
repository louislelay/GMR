"""Tests for the built-in BVH reader."""

from pathlib import Path

import numpy as np

from general_motion_retargeting.sources.bvh import load_bvh_motion

_BVH = """\
HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftFoot
  {
    OFFSET 0 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftToe
    {
      OFFSET 0 10 0
      CHANNELS 3 Zrotation Xrotation Yrotation
    }
  }
  JOINT RightFoot
  {
    OFFSET 0 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightToe
    {
      OFFSET 0 -10 0
      CHANNELS 3 Zrotation Xrotation Yrotation
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""


def test_loads_and_resamples_bvh(tmp_path: Path) -> None:
    path = tmp_path / "motion.bvh"
    path.write_text(_BVH, encoding="utf-8")

    motion = load_bvh_motion(path, target_fps=20.0)

    assert motion.source_fps == 10.0
    assert motion.fps == 20.0
    assert len(motion.frames) == 4
    np.testing.assert_allclose(motion.frames[-1]["Hips"][0], [1.0, 0.0, 0.0])
    assert "LeftFootMod" in motion.frames[0]
    assert "RightFootMod" in motion.frames[0]
