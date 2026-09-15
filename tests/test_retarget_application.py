"""Tests for retarget file and directory orchestration."""

from pathlib import Path

import pytest
from synthetic_motion import build_frames

from general_motion_retargeting import HumanMotion, load_robot_motion
from general_motion_retargeting.application import RetargetApplication
from general_motion_retargeting.catalog import build_catalog
from general_motion_retargeting.cli import retarget as retarget_cli


class _SyntheticApplication(RetargetApplication):
    def load_source(self, path: Path, *, source: str, target_fps: float) -> HumanMotion:
        del path
        return HumanMotion(
            frames=tuple(build_frames()[:2]),
            fps=target_fps,
            height=1.8,
            source_format=source,
            source_identifier="sha256:test",
            source_fps=60.0,
        )


def test_retarget_file_uses_public_service_and_canonical_npz(
    tmp_path: Path,
) -> None:
    output = tmp_path / "robot.npz"
    application = _SyntheticApplication(catalog=build_catalog(discover=False))

    motion = application.retarget_file(
        tmp_path / "source.npz",
        output,
        source="smplx",
        robot="unitree_g1",
    )

    assert motion.frame_count == 2
    assert motion.source_identifier == "sha256:test"
    assert load_robot_motion(output).robot == "unitree_g1"


def test_cli_mirrors_directory_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "nested" / "walk.bvh"
    source.parent.mkdir(parents=True)
    source.touch()
    outputs: list[Path] = []

    class FakeApplication:
        def __init__(self, *, body_models: Path | None = None) -> None:
            del body_models

        def retarget_file(
            self,
            input_path: Path,
            output_path: Path,
            **kwargs: object,
        ) -> None:
            del input_path, kwargs
            output_path.parent.mkdir(parents=True)
            output_path.touch()
            outputs.append(output_path)

    monkeypatch.setattr(retarget_cli, "RetargetApplication", FakeApplication)

    status = retarget_cli.main(
        [
            "--source",
            "bvh",
            "--robot",
            "unitree_g1",
            "--input",
            str(input_root),
            "--output",
            str(output_root),
        ]
    )

    assert status == 0
    assert outputs == [output_root / "nested" / "walk.npz"]
