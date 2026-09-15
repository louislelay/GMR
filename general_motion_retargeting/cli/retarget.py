"""Retarget source files into canonical robot motions."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from rich import print

from ..application import RetargetApplication
from ..catalog import build_catalog


def _parser() -> argparse.ArgumentParser:
    catalog = build_catalog()
    parser = argparse.ArgumentParser(
        prog="retarget",
        description="Retarget SMPL-X or BVH motion into canonical GMR NPZ.",
    )
    parser.add_argument("--robot", required=True, choices=sorted(catalog.robots))
    parser.add_argument(
        "--source", required=True, choices=("smplx", "bvh", "bvh_nokov")
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--body-models", type=Path)
    parser.add_argument("--offset-to-ground", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _input_files(path: Path, source: str) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise ValueError(f"input does not exist: {path}")
    suffix = ".npz" if source == "smplx" else ".bvh"
    files = tuple(sorted(path.rglob(f"*{suffix}")))
    if not files:
        raise ValueError(f"input directory contains no {suffix} files: {path}")
    return files


def _output_path(input_root: Path, output_root: Path, source: Path) -> Path:
    if input_root.is_file():
        return output_root
    return output_root / source.relative_to(input_root).with_suffix(".npz")


def main(argv: Sequence[str] | None = None) -> int:
    """Run file or directory retargeting.

    Args:
        argv: Optional command arguments excluding the executable name.

    Returns:
        Process exit status.
    """
    args = _parser().parse_args(argv)
    body_models = args.body_models
    if body_models is None and os.environ.get("GMR_SMPLX_MODELS"):
        body_models = Path(os.environ["GMR_SMPLX_MODELS"])
    application = RetargetApplication(body_models=body_models)
    files = _input_files(args.input, args.source)
    if args.input.is_dir() and args.output.suffix:
        raise ValueError("directory input requires an output directory")
    for input_path in files:
        output_path = _output_path(args.input, args.output, input_path)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"output exists: {output_path}; pass --overwrite to replace"
            )
        application.retarget_file(
            input_path,
            output_path,
            source=args.source,
            robot=args.robot,
            target_fps=args.fps,
            offset_to_ground=args.offset_to_ground,
        )
        print(f"[green]Saved[/green] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
