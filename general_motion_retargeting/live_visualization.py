"""Mjviser consumer for reusable retargeted streams."""

import time

import mjviser
import mujoco as mj
import numpy as np
import viser

from .models import RetargetingProfile, RobotSpec
from .streaming import FrameSubscription, RetargetedFrame

_SOURCE_GROUP = 3
_ROBOT_GROUP = 4


class LiveWorkspace:
    """Latest-value mjviser consumer for a retargeted stream."""

    def __init__(
        self,
        *,
        robot: RobotSpec,
        profile: RetargetingProfile,
        subscription: FrameSubscription,
        fps: float,
        server: viser.ViserServer | None = None,
    ) -> None:
        """Create the live browser workspace.

        Args:
            robot: Robot scene provider.
            profile: Active retargeting profile.
            subscription: Latest-value viewer subscription.
            fps: Viewer update rate.
            server: Optional injected Viser server.
        """
        self.subscription = subscription
        self._last_frame: RetargetedFrame | None = None
        specification = robot.assets.load_scene_spec()
        for keyframe in list(specification.keys):
            specification.delete(keyframe)
        self._source_names = tuple(
            sorted({match.human.body for stage in profile.stages for match in stage})
        )
        for index, _ in enumerate(self._source_names):
            body = specification.worldbody.add_body(
                name=f"_gmr_live_source_{index}", mocap=True
            )
            body.add_site(
                name=f"_gmr_live_source_site_{index}",
                size=[0.012, 0.0, 0.0],
                rgba=[1.0, 0.55, 0.0, 1.0],
                group=_SOURCE_GROUP,
            )
        seen: set[str] = set()
        for stage in profile.stages:
            for match in stage:
                if match.identifier in seen:
                    continue
                seen.add(match.identifier)
                specification.body(match.robot.body).add_site(
                    name=f"_gmr_live_robot_{len(seen)}",
                    pos=match.robot.position,
                    quat=match.robot.rotation,
                    size=[0.012, 0.0, 0.0],
                    rgba=[0.0, 1.0, 0.0, 1.0],
                    group=_ROBOT_GROUP,
                )
        self.model = specification.compile()
        self.data = mj.MjData(self.model)
        self.model.opt.timestep = 1.0 / fps
        self._mocap_ids = tuple(
            int(
                self.model.body_mocapid[self.model.body(f"_gmr_live_source_{index}").id]
            )
            for index in range(len(self._source_names))
        )
        self.server = server or viser.ViserServer()
        with self.server.gui.add_folder("GMR Stream"):
            self._status = self.server.gui.add_html("Waiting for frames…")
            source_sites = self.server.gui.add_checkbox(
                "Mocap reference sites", initial_value=True
            )
            robot_sites = self.server.gui.add_checkbox(
                "Robot target sites", initial_value=False
            )
        self.viewer = mjviser.Viewer(
            self.model,
            self.data,
            step_fn=self._step,
            render_fn=self._render,
            server=self.server,
        )

        @source_sites.on_update
        def _(_) -> None:
            self.viewer.scene.site_groups_visible[_SOURCE_GROUP] = source_sites.value
            self.viewer.scene.refresh_visualization()

        @robot_sites.on_update
        def _(_) -> None:
            self.viewer.scene.site_groups_visible[_ROBOT_GROUP] = robot_sites.value
            self.viewer.scene.refresh_visualization()

    def _apply(self, frame: RetargetedFrame) -> None:
        if frame.qpos.shape != (self.model.nq,):
            raise ValueError(
                f"stream qpos has shape {frame.qpos.shape}, "
                f"scene expects {(self.model.nq,)}"
            )
        self.data.qpos[:] = frame.qpos
        self.data.mocap_pos[list(self._mocap_ids)] = np.stack(
            [frame.source_frame[name][0] for name in self._source_names]
        )
        mj.mj_forward(self.model, self.data)
        pipeline_ms = (frame.completed_monotonic - frame.received_monotonic) * 1000.0
        age_ms = (time.monotonic() - frame.completed_monotonic) * 1000.0
        self._status.content = (
            f"<strong>Sequence:</strong> {frame.sequence}<br/>"
            f"<strong>Retarget:</strong> {pipeline_ms:.1f} ms<br/>"
            f"<strong>Local age:</strong> {age_ms:.1f} ms<br/>"
            f"<strong>Viewer drops:</strong> {self.subscription.dropped_count}"
        )

    def _step(self, model: mj.MjModel, data: mj.MjData) -> None:
        del model, data
        frame = self.subscription.read_latest()
        if frame is None:
            return
        self._last_frame = frame
        self._apply(frame)

    def _render(self, scene: mjviser.ViserMujocoScene) -> None:
        scene.update_from_mjdata(self.data)

    def run(self) -> None:
        """Run the live workspace until interrupted."""
        self.viewer.run()

    def close(self) -> None:
        """Stop the Viser server."""
        self.server.stop()
