"""Mjviser motion workspace with synchronized source and robot playback."""

import threading
from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio.v2 as imageio
import mjviser
import mujoco as mj
import numpy as np
import viser

from .models import HumanMotion, RetargetingProfile, RobotMotion, RobotSpec
from .motion_io import save_robot_motion

_SOURCE_GROUP = 3
_ROBOT_SITE_GROUP = 4
_SITE_SIZE = 0.012
_SIDE_OFFSET = 0.75
_EMPTY_SCENE = """
<mujoco model="gmr_source">
  <statistic meansize="0.1" extent="2"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="ground" type="plane" size="5 5 0.05" rgba=".2 .3 .4 1"/>
  </worldbody>
</mujoco>
"""
_SKELETON_EDGES = (
    ("pelvis", "spine3"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
    ("spine3", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("spine3", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("Hips", "Spine1"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftToeBase"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightToeBase"),
    ("Spine1", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
)


class MotionWorkspace:
    """Interactive mjviser workspace for source and robot motions."""

    def __init__(
        self,
        *,
        robot: RobotSpec | None,
        robot_motion: RobotMotion | None,
        source_motion: HumanMotion | None = None,
        profile: RetargetingProfile | None = None,
        source_time_offset: float = 0.0,
        server: viser.ViserServer | None = None,
    ) -> None:
        """Create a synchronized browser workspace.

        Args:
            robot: Robot scene provider when robot motion is present.
            robot_motion: Optional canonical robot motion.
            source_motion: Optional source motion and SMPL-X mesh.
            profile: Optional profile used to place robot target sites.
            source_time_offset: Source timeline offset relative to robot seconds.
            server: Optional injected Viser server.
        """
        if robot_motion is None and source_motion is None:
            raise ValueError("visualize requires robot motion, source motion, or both")
        if robot_motion is not None and robot is None:
            raise ValueError("robot motion requires a robot specification")
        if robot_motion is not None and robot_motion.robot != robot.identifier:
            raise ValueError(
                f"motion targets {robot_motion.robot!r}, not {robot.identifier!r}"
            )
        if profile is not None and robot is not None:
            if profile.robot != robot.identifier:
                raise ValueError(
                    f"profile targets {profile.robot!r}, not {robot.identifier!r}"
                )
        self.robot = robot
        self.robot_motion = robot_motion
        self.source_motion = source_motion
        self.profile = profile
        self.source_time_offset = source_time_offset
        self._lock = threading.Lock()
        self._time = 0.0
        self._exporting = False
        specification = (
            robot.assets.load_scene_spec()
            if robot is not None
            else mj.MjSpec.from_string(_EMPTY_SCENE)
        )
        for keyframe in list(specification.keys):
            specification.delete(keyframe)
        robot_sites = self._inject_robot_sites(specification)
        self._robot_source_names = tuple(item[0] for item in robot_sites)
        self._robot_site_names = tuple(item[1] for item in robot_sites)
        self._source_names = self._inject_source_sites(specification)
        self.model = specification.compile()
        self.data = mj.MjData(self.model)
        self.model.opt.timestep = 1.0 / self._maximum_fps()
        self._joint_addresses = self._resolve_joint_addresses()
        self._source_mocap_ids = self._resolve_source_mocap_ids()
        self._duration = self._playback_duration()
        self.server = server or viser.ViserServer()
        self.viewer = mjviser.Viewer(
            self.model,
            self.data,
            step_fn=self._step,
            render_fn=self._render,
            reset_fn=self._reset,
            server=self.server,
        )
        self._human_mesh: viser.MeshHandle | None = None
        self._source_skeleton: viser.LineSegmentsHandle | None = None
        self._robot_skeleton: viser.LineSegmentsHandle | None = None
        self._setup_overlays()
        self._setup_controls()
        self._apply_time(0.0)

    def _inject_robot_sites(
        self, specification: mj.MjSpec
    ) -> tuple[tuple[str, str], ...]:
        if self.profile is None or self.robot is None:
            return ()
        sites: list[tuple[str, str]] = []
        seen: set[str] = set()
        for stage in self.profile.stages:
            for match in stage:
                if match.identifier in seen:
                    continue
                seen.add(match.identifier)
                name = f"_gmr_robot_{len(sites)}"
                parent = specification.body(match.robot.body)
                if parent is None:
                    raise ValueError(
                        f"scene is missing profile body {match.robot.body!r}"
                    )
                parent.add_site(
                    name=name,
                    pos=match.robot.position,
                    quat=match.robot.rotation,
                    size=[_SITE_SIZE, 0.0, 0.0],
                    rgba=[0.0, 1.0, 0.0, 1.0],
                    group=_ROBOT_SITE_GROUP,
                )
                sites.append((match.human.body, name))
        return tuple(sites)

    def _inject_source_sites(self, specification: mj.MjSpec) -> tuple[str, ...]:
        if self.source_motion is None:
            return ()
        names = tuple(sorted(self.source_motion.frames[0]))
        for index, _ in enumerate(names):
            body = specification.worldbody.add_body(
                name=f"_gmr_source_{index}", mocap=True
            )
            body.add_site(
                name=f"_gmr_source_site_{index}",
                size=[_SITE_SIZE, 0.0, 0.0],
                rgba=[1.0, 0.55, 0.0, 1.0],
                group=_SOURCE_GROUP,
            )
        return names

    def _resolve_joint_addresses(self) -> tuple[int, ...]:
        if self.robot_motion is None:
            return ()
        addresses: list[int] = []
        for name in self.robot_motion.joint_names:
            joint = self.model.joint(name)
            if joint.id < 0:
                raise ValueError(f"scene is missing motion joint {name!r}")
            addresses.append(int(joint.qposadr[0]))
        return tuple(addresses)

    def _resolve_source_mocap_ids(self) -> tuple[int, ...]:
        return tuple(
            int(self.model.body_mocapid[self.model.body(f"_gmr_source_{index}").id])
            for index in range(len(self._source_names))
        )

    def _maximum_fps(self) -> float:
        rates = [
            motion.fps
            for motion in (self.robot_motion, self.source_motion)
            if motion is not None
        ]
        return max(rates)

    def _playback_duration(self) -> float:
        durations = [
            motion.duration
            if isinstance(motion, RobotMotion)
            else len(motion.frames) / motion.fps
            for motion in (self.robot_motion, self.source_motion)
            if motion is not None
        ]
        return min(durations)

    def _robot_frame(self, time_seconds: float) -> int:
        if self.robot_motion is None:
            return 0
        return min(
            int(time_seconds * self.robot_motion.fps),
            self.robot_motion.frame_count - 1,
        )

    def _source_frame(self, time_seconds: float) -> int:
        if self.source_motion is None:
            return 0
        source_time = max(0.0, time_seconds - self.source_time_offset)
        return min(
            int(source_time * self.source_motion.fps),
            len(self.source_motion.frames) - 1,
        )

    def _apply_time(self, time_seconds: float) -> None:
        self._time = min(max(time_seconds, 0.0), self._duration)
        if self.robot_motion is not None:
            frame = self._robot_frame(self._time)
            self.data.qpos[:3] = self.robot_motion.root_positions[frame]
            self.data.qpos[3:7] = self.robot_motion.root_quaternions[frame]
            self.data.qpos[list(self._joint_addresses)] = (
                self.robot_motion.joint_positions[frame]
            )
            if self.source_motion is not None:
                self.data.qpos[1] -= _SIDE_OFFSET
        if self.source_motion is not None:
            frame_data = self.source_motion.frames[self._source_frame(self._time)]
            positions = np.stack([frame_data[name][0] for name in self._source_names])
            if self.robot_motion is not None:
                positions[:, 1] += _SIDE_OFFSET
            self.data.mocap_pos[list(self._source_mocap_ids)] = positions
        mj.mj_forward(self.model, self.data)

    def _skeleton_points(self, source: bool) -> np.ndarray:
        names = self._source_names if source else self._robot_source_names
        name_to_index = {name: index for index, name in enumerate(names)}
        edges = tuple(
            (name_to_index[start], name_to_index[end])
            for start, end in _SKELETON_EDGES
            if start in name_to_index and end in name_to_index
        )
        if not edges:
            return np.empty((0, 2, 3), dtype=np.float64)
        if source:
            frame = self.source_motion.frames[self._source_frame(self._time)]
            positions = np.stack([frame[name][0] for name in self._source_names])
            if self.robot_motion is not None:
                positions[:, 1] += _SIDE_OFFSET
            return positions[np.asarray(edges)]
        site_positions = np.stack(
            [self.data.site(name).xpos for name in self._robot_site_names]
        )
        return site_positions[np.asarray(edges)]

    def _setup_overlays(self) -> None:
        if (
            self.source_motion is not None
            and self.source_motion.body_vertices is not None
            and self.source_motion.body_faces is not None
        ):
            self._human_mesh = self.server.scene.add_mesh_simple(
                "/gmr/source/body",
                self._human_vertices(),
                self.source_motion.body_faces,
                color=(120, 170, 255),
                opacity=0.85,
                side="double",
            )
        if self.source_motion is not None:
            self._source_skeleton = self.server.scene.add_line_segments(
                "/gmr/source/skeleton",
                self._skeleton_points(source=True),
                colors=(255, 150, 0),
                thickness=0.005,
                visible=False,
            )
        if self._robot_site_names and self.source_motion is not None:
            points = self._skeleton_points(source=False)
            if len(points):
                self._robot_skeleton = self.server.scene.add_line_segments(
                    "/gmr/robot/skeleton",
                    points,
                    colors=(0, 255, 0),
                    thickness=0.005,
                    visible=False,
                )

    def _human_vertices(self) -> np.ndarray:
        vertices = self.source_motion.body_vertices[
            self._source_frame(self._time)
        ].copy()
        if self.robot_motion is not None:
            vertices[:, 1] += _SIDE_OFFSET
        return vertices

    def _set_site_visibility(self, group: int, visible: bool) -> None:
        self.viewer.scene.site_groups_visible[group] = visible
        self.viewer.scene.refresh_visualization()

    def _setup_controls(self) -> None:
        with self.server.gui.add_folder("GMR Motion"):
            timeline = self.server.gui.add_slider(
                "Time (s)",
                min=0.0,
                max=self._duration,
                step=1.0 / self._maximum_fps(),
                initial_value=0.0,
            )
            trim = self.server.gui.add_multi_slider(
                "Trim (s)",
                min=0.0,
                max=self._duration,
                step=1.0 / self._maximum_fps(),
                initial_value=(0.0, self._duration),
                min_range=1.0 / self._maximum_fps(),
            )
            source_sites = self.server.gui.add_checkbox(
                "Mocap reference sites",
                initial_value=False,
                disabled=self.source_motion is None,
            )
            robot_sites = self.server.gui.add_checkbox(
                "Robot target sites",
                initial_value=False,
                disabled=not self._robot_site_names,
            )
            source_skeleton = self.server.gui.add_checkbox(
                "Source skeleton",
                initial_value=False,
                disabled=self._source_skeleton is None,
            )
            robot_skeleton = self.server.gui.add_checkbox(
                "Robot skeleton",
                initial_value=False,
                disabled=self._robot_skeleton is None,
            )
            download = self.server.gui.add_button(
                "Download trimmed NPZ", disabled=self.robot_motion is None
            )
            render = self.server.gui.add_button("Render selected range to video")

        @timeline.on_update
        def _(_) -> None:
            with self._lock:
                self._apply_time(float(timeline.value))
                self._render(self.viewer.scene)

        @source_sites.on_update
        def _(_) -> None:
            self._set_site_visibility(_SOURCE_GROUP, source_sites.value)

        @robot_sites.on_update
        def _(_) -> None:
            self._set_site_visibility(_ROBOT_SITE_GROUP, robot_sites.value)

        @source_skeleton.on_update
        def _(_) -> None:
            self._source_skeleton.visible = source_skeleton.value

        @robot_skeleton.on_update
        def _(_) -> None:
            self._robot_skeleton.visible = robot_skeleton.value

        @download.on_click
        def _(event) -> None:
            if event.client is None or self.robot_motion is None:
                return
            start, end = (float(value) for value in trim.value)
            with NamedTemporaryFile(suffix=".npz") as temporary:
                path = Path(temporary.name)
                save_robot_motion(path, self.robot_motion.trim(start, end))
                event.client.send_file_download("trimmed_motion.npz", path.read_bytes())

        @render.on_click
        def _(event) -> None:
            if event.client is None:
                return
            self._render_video(event.client, tuple(float(v) for v in trim.value))

    def _render_video(
        self, client: viser.ClientHandle, bounds: tuple[float, float]
    ) -> None:
        start, end = bounds
        frame_times = np.arange(start, end, 1.0 / self._maximum_fps(), dtype=np.float64)
        progress = self.server.gui.add_progress_bar(0.0)
        self._exporting = True
        try:
            with NamedTemporaryFile(suffix=".mp4") as temporary:
                with imageio.get_writer(
                    temporary.name, fps=self._maximum_fps()
                ) as writer:
                    for index, time_seconds in enumerate(frame_times):
                        with self._lock:
                            self._apply_time(float(time_seconds))
                            self._render(self.viewer.scene)
                        writer.append_data(
                            client.get_render(height=720, width=1280, timeout=10.0)
                        )
                        progress.value = (index + 1) / len(frame_times)
                client.send_file_download(
                    "gmr_motion.mp4", Path(temporary.name).read_bytes()
                )
        finally:
            self._exporting = False
            progress.remove()

    def _render(self, scene: mjviser.ViserMujocoScene) -> None:
        scene.update_from_mjdata(self.data)
        if self._human_mesh is not None:
            self._human_mesh.vertices = self._human_vertices()
        if self._source_skeleton is not None:
            self._source_skeleton.points = self._skeleton_points(source=True)
        if self._robot_skeleton is not None:
            self._robot_skeleton.points = self._skeleton_points(source=False)

    def _step(self, model: mj.MjModel, data: mj.MjData) -> None:
        del model, data
        if self._exporting:
            return
        with self._lock:
            self._apply_time((self._time + self.model.opt.timestep) % self._duration)

    def _reset(self, model: mj.MjModel, data: mj.MjData) -> None:
        del model, data
        with self._lock:
            self._apply_time(0.0)

    def run(self) -> None:
        """Run the workspace until interrupted."""
        self.viewer.run()

    def close(self) -> None:
        """Stop the injected or internally-created Viser server."""
        self.server.stop()
