"""Application services shared by the Python API and command line."""

import hashlib
from pathlib import Path

from .catalog import Catalog, build_catalog
from .models import HumanMotion, RobotMotion
from .motion_io import save_robot_motion
from .retargeter import Retargeter
from .sources.bvh import load_bvh_motion


def _source_identifier(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _normalized_source(source: str) -> str:
    if source == "bvh":
        return "bvh_lafan1"
    return source


class RetargetApplication:
    """Retarget files through injected catalog and source-model resources."""

    catalog: Catalog
    body_models: Path | None

    def __init__(
        self,
        *,
        catalog: Catalog | None = None,
        body_models: Path | None = None,
    ) -> None:
        """Initialize file retargeting resources."""
        self.catalog = catalog or build_catalog()
        self.body_models = body_models

    def load_source(self, path: Path, *, source: str, target_fps: float) -> HumanMotion:
        """Load one supported source motion."""
        normalized = _normalized_source(source)
        if normalized == "smplx":
            return self._load_smplx(path, target_fps)
        if normalized.startswith("bvh_"):
            return self._load_bvh(path, normalized, target_fps)
        raise ValueError(f"file source {source!r} is not supported; use smplx or bvh")

    def _load_smplx(self, path: Path, target_fps: float) -> HumanMotion:
        if self.body_models is None:
            raise ValueError(
                "SMPL-X retargeting requires body_models or GMR_SMPLX_MODELS"
            )
        from .sources.smplx import load_smplx_motion

        data = load_smplx_motion(path, self.body_models, target_fps=target_fps)
        return HumanMotion(
            frames=data.frames,
            fps=data.fps,
            height=data.height,
            source_format="smplx",
            source_identifier=_source_identifier(path),
            source_fps=data.source_fps,
            body_vertices=data.vertices,
            body_faces=data.faces,
        )

    def _load_bvh(self, path: Path, source: str, target_fps: float) -> HumanMotion:
        convention = source.removeprefix("bvh_")
        if convention not in {"lafan1", "nokov"}:
            raise ValueError(f"unsupported BVH convention: {convention}")
        data = load_bvh_motion(
            path,
            convention=convention,
            target_fps=target_fps,
        )
        return HumanMotion(
            frames=data.frames,
            fps=data.fps,
            height=data.height,
            source_format=source,
            source_identifier=_source_identifier(path),
            source_fps=data.source_fps,
        )

    def retarget_file(
        self,
        input_path: Path,
        output_path: Path,
        *,
        source: str,
        robot: str,
        target_fps: float = 30.0,
        offset_to_ground: bool = False,
    ) -> RobotMotion:
        """Retarget one source file and save canonical NPZ."""
        normalized = _normalized_source(source)
        human_motion = self.load_source(
            input_path, source=normalized, target_fps=target_fps
        )
        retargeter = Retargeter(
            self.catalog.robot(robot),
            self.catalog.profile(normalized, robot),
        )
        robot_motion = retargeter.retarget(
            human_motion, offset_to_ground=offset_to_ground
        )
        save_robot_motion(output_path, robot_motion)
        return robot_motion
