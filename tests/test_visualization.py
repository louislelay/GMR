"""Smoke tests for the mjviser motion workspace."""

import viser
from synthetic_motion import build_frames

from general_motion_retargeting import HumanMotion
from general_motion_retargeting.visualization import MotionWorkspace


def test_source_only_workspace_uses_mjviser() -> None:
    motion = HumanMotion(
        frames=tuple(build_frames()[:2]),
        fps=30.0,
        height=1.8,
        source_format="smplx",
    )
    server = viser.ViserServer(port=0, verbose=False)
    workspace = MotionWorkspace(
        robot=None,
        robot_motion=None,
        source_motion=motion,
        server=server,
    )

    try:
        assert workspace.model.nmocap == len(motion.frames[0])
        assert workspace.viewer.scene is not None
    finally:
        workspace.close()
