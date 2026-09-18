"""Retarget and visualize a live source."""

import argparse
from collections.abc import Sequence

from ..catalog import build_catalog
from ..live_visualization import LiveWorkspace
from ..retargeter import Retargeter
from ..sources import SourceOptions, create_live_source
from ..streaming import RetargetedStream


def _parser() -> argparse.ArgumentParser:
    catalog = build_catalog()
    parser = argparse.ArgumentParser(
        prog="stream",
        description="Retarget a live source and attach an mjviser consumer.",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--robot", required=True, choices=sorted(catalog.robots))
    parser.add_argument("--fps", type=float)
    parser.add_argument("--port", type=int)
    parser.add_argument("--human-height", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run live retargeting with an mjviser latest-value consumer."""
    args = _parser().parse_args(argv)
    catalog = build_catalog()
    source = create_live_source(
        args.source,
        SourceOptions(
            fps=args.fps,
            port=args.port,
            human_height=args.human_height,
        ),
    )
    profile = catalog.profile(source.source_format, args.robot)
    stream = RetargetedStream(
        source,
        Retargeter(catalog.robot(args.robot), profile),
    )
    subscription = stream.subscribe()
    workspace = LiveWorkspace(
        robot=catalog.robot(args.robot),
        profile=profile,
        subscription=subscription,
        fps=args.fps or source.fps,
    )
    try:
        with stream:
            workspace.run()
    finally:
        workspace.close()
    if stream.error is not None:
        raise RuntimeError("live retargeting worker failed") from stream.error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
