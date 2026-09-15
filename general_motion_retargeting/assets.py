"""Robot asset providers."""

from dataclasses import dataclass
from pathlib import Path

import mujoco as mj


@dataclass(frozen=True)
class PathRobotAssets:
    """Load robot specifications from local MJCF files."""

    model_path: Path
    scene_path: Path | None = None

    def load_model_spec(self) -> mj.MjSpec:
        """Load the editable robot-only specification."""
        return mj.MjSpec.from_file(str(self.model_path))

    def load_scene_spec(self) -> mj.MjSpec:
        """Load the editable visualization specification."""
        return mj.MjSpec.from_file(str(self.scene_path or self.model_path))
