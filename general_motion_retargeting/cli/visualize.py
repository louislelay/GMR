"""Open the mjviser motion workspace."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from ..application import RetargetApplication
from ..catalog import build_catalog
from ..motion_io import load_robot_motion
from ..visualization import MotionWorkspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize",
        description="Inspect, trim, and render GMR motion in mjviser.",
    )
    parser.add_argument("--robot-motion", type=Path)
    parser.add_argument("--source-motion", type=Path)
    parser.add_argument("--source", choices=("smplx", "bvh", "bvh_nokov"))
    parser.add_argument("--robot")
    parser.add_argument("--body-models", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--source-offset", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive mjviser workspace."""
    args = _parser().parse_args(argv)
    if args.robot_motion is None and args.source_motion is None:
        raise ValueError("provide --robot-motion, --source-motion, or both")
    robot_motion = (
        load_robot_motion(args.robot_motion) if args.robot_motion is not None else None
    )
    robot_identifier = args.robot
    if robot_motion is not None:
        if robot_identifier is not None and robot_identifier != robot_motion.robot:
            raise ValueError("--robot does not match robot-motion metadata")
        robot_identifier = robot_motion.robot
    source = args.source
    if source is None and robot_motion is not None:
        source = robot_motion.source_format
    if args.source_motion is not None and source is None:
        raise ValueError("--source is required with --source-motion")
    body_models = args.body_models
    if body_models is None and os.environ.get("GMR_SMPLX_MODELS"):
        body_models = Path(os.environ["GMR_SMPLX_MODELS"])
    catalog = build_catalog()
    source_motion = None
    if args.source_motion is not None:
        source_motion = RetargetApplication(
            catalog=catalog, body_models=body_models
        ).load_source(args.source_motion, source=source, target_fps=args.fps)
    robot = catalog.robot(robot_identifier) if robot_identifier is not None else None
    profile = None
    if robot_identifier is not None and source is not None:
        normalized_source = "bvh_lafan1" if source == "bvh" else source
        profile = catalog.profile(normalized_source, robot_identifier)
    workspace = MotionWorkspace(
        robot=robot,
        robot_motion=robot_motion,
        source_motion=source_motion,
        profile=profile,
        source_time_offset=args.source_offset,
    )
    workspace.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
