"""Injected, typed motion-retargeting session."""

from dataclasses import dataclass

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .models import (
    FloatArray,
    HumanFrame,
    HumanMotion,
    Match,
    RetargetingProfile,
    RobotMotion,
    RobotSpec,
    SolverSettings,
)
from .profiles import validate_human_frame

_GROUND_CLEARANCE = 0.1


@dataclass(frozen=True)
class _Task:
    match: Match
    frame: mink.FrameTask


@dataclass(frozen=True)
class _SiteMatch:
    match: Match
    site_name: str


class Retargeter:
    """Stateful retargeting session for one robot and profile."""

    def __init__(
        self,
        robot: RobotSpec,
        profile: RetargetingProfile,
        settings: SolverSettings | None = None,
    ) -> None:
        """Initialize the MuJoCo and Mink session.

        Args:
            robot: Target robot description and model provider.
            profile: Source-to-robot retargeting parameters.
            settings: Optional solver and convergence settings.
        """
        if profile.robot != robot.identifier:
            raise ValueError(
                f"profile targets {profile.robot!r}, not {robot.identifier!r}"
            )
        self.robot = robot
        self.profile = profile
        self.settings = settings or SolverSettings()
        specification = robot.assets.load_model_spec()
        site_stages = tuple(
            self._inject_sites(specification, stage, stage_index)
            for stage_index, stage in enumerate(profile.stages)
        )
        self.model = specification.compile()
        self.configuration = mink.Configuration(self.model)
        self._validate_model_frames()
        self._stages = tuple(self._build_stage(stage) for stage in site_stages)
        self._limits: list[mink.Limit] = [mink.ConfigurationLimit(self.model)]
        if self.settings.use_velocity_limits:
            velocities = {
                name: self.settings.velocity_limit for name in self._joint_names()
            }
            self._limits.append(mink.VelocityLimit(self.model, velocities))
        self._required_bodies = frozenset(
            {profile.human_root}
            | {task.match.human.body for stage in self._stages for task in stage}
        )

    def _validate_model_frames(self) -> None:
        body_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, index)
            for index in range(self.model.nbody)
        }
        if self.robot.root_body not in body_names:
            raise ValueError(
                f"robot root body {self.robot.root_body!r} is missing from model"
            )

    def _joint_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
                continue
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise ValueError(f"robot joint {joint_id} has no name")
            names.append(name)
        return tuple(names)

    def _inject_sites(
        self,
        specification: mj.MjSpec,
        matches: tuple[Match, ...],
        stage_index: int,
    ) -> tuple[_SiteMatch, ...]:
        site_matches: list[_SiteMatch] = []
        for match_index, match in enumerate(matches):
            if match.position_weight == 0.0 and match.orientation_weight == 0.0:
                continue
            parent = specification.body(match.robot.body)
            if parent is None:
                raise ValueError(
                    f"robot model is missing profile body {match.robot.body!r}"
                )
            site_name = f"_gmr_{stage_index}_{match_index}"
            parent.add_site(
                name=site_name,
                pos=match.robot.position,
                quat=match.robot.rotation,
            )
            site_matches.append(_SiteMatch(match=match, site_name=site_name))
        return tuple(site_matches)

    def _build_stage(self, site_matches: tuple[_SiteMatch, ...]) -> tuple[_Task, ...]:
        tasks: list[_Task] = []
        for site_match in site_matches:
            match = site_match.match
            tasks.append(
                _Task(
                    match=match,
                    frame=mink.FrameTask(
                        frame_name=site_match.site_name,
                        frame_type="site",
                        position_cost=match.position_weight,
                        orientation_cost=match.orientation_weight,
                        lm_damping=1.0,
                    ),
                )
            )
        return tuple(tasks)

    def _scaled_frame(
        self, frame: HumanFrame, human_height: float | None
    ) -> dict[str, tuple[FloatArray, FloatArray]]:
        validate_human_frame(frame, self._required_bodies)
        ratio = 1.0
        if human_height is not None:
            ratio = human_height / self.profile.human_height_assumption
        root_position, root_rotation = frame[self.profile.human_root]
        root_scale = self.profile.human_scales[self.profile.human_root] * ratio
        scaled_root = np.asarray(root_position, dtype=np.float64) * root_scale
        scaled = {
            self.profile.human_root: (
                scaled_root,
                np.asarray(root_rotation, dtype=np.float64),
            )
        }
        for body_name, scale in self.profile.human_scales.items():
            if body_name == self.profile.human_root:
                continue
            position, rotation = frame[body_name]
            scaled[body_name] = (
                (np.asarray(position) - root_position) * scale * ratio + scaled_root,
                np.asarray(rotation, dtype=np.float64),
            )
        return scaled

    def _offset_to_ground(
        self, frame: dict[str, tuple[FloatArray, FloatArray]]
    ) -> dict[str, tuple[FloatArray, FloatArray]]:
        heights = [
            float(position[2])
            for name, (position, _) in frame.items()
            if "foot" in name.lower()
        ]
        if not heights:
            raise ValueError("offset_to_ground requires a human body containing 'foot'")
        shift = np.array([0.0, 0.0, _GROUND_CLEARANCE - min(heights)], dtype=np.float64)
        return {
            name: (position + shift, rotation)
            for name, (position, rotation) in frame.items()
        }

    def _set_targets(
        self,
        tasks: tuple[_Task, ...],
        frame: dict[str, tuple[FloatArray, FloatArray]],
    ) -> None:
        ground = self.profile.ground_height * np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        for task in tasks:
            position, quaternion = frame[task.match.human.body]
            rotation = Rotation.from_quat(
                quaternion, scalar_first=True
            ) * Rotation.from_quat(task.match.human.rotation, scalar_first=True)
            target_position = position + rotation.apply(
                task.match.human.position - ground
            )
            task.frame.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(rotation.as_quat(scalar_first=True)),
                    target_position,
                )
            )

    def _error(self, tasks: tuple[_Task, ...]) -> float:
        if not tasks:
            return 0.0
        errors = [task.frame.compute_error(self.configuration) for task in tasks]
        return float(np.linalg.norm(np.concatenate(errors)))

    def _solve(self, tasks: tuple[_Task, ...]) -> None:
        if not tasks:
            return
        current_error = self._error(tasks)
        timestep = self.model.opt.timestep
        for _ in range(self.settings.max_iterations + 1):
            velocity = mink.solve_ik(
                self.configuration,
                [task.frame for task in tasks],
                timestep,
                self.settings.solver,
                self.settings.damping,
                limits=self._limits,
            )
            self.configuration.integrate_inplace(velocity, timestep)
            next_error = self._error(tasks)
            if current_error - next_error <= self.settings.improvement_threshold:
                return
            current_error = next_error

    @property
    def qpos(self) -> FloatArray:
        """Return a copy of the current MuJoCo qpos."""
        return np.asarray(self.configuration.data.qpos.copy(), dtype=np.float64)

    def retarget_frame(
        self,
        frame: HumanFrame,
        *,
        human_height: float | None = None,
        offset_to_ground: bool = False,
    ) -> FloatArray:
        """Retarget one frame while preserving solver state.

        Args:
            frame: Global human body positions and wxyz orientations.
            human_height: Source subject height in meters, when known.
            offset_to_ground: Shift the lowest human foot to ground clearance.

        Returns:
            Copied robot qpos.
        """
        scaled = self._scaled_frame(frame, human_height)
        if offset_to_ground:
            scaled = self._offset_to_ground(scaled)
        for stage in self._stages:
            self._set_targets(stage, scaled)
            self._solve(stage)
        return self.qpos

    def retarget(
        self, motion: HumanMotion, *, offset_to_ground: bool = False
    ) -> RobotMotion:
        """Retarget a complete human motion.

        Args:
            motion: Typed source motion.
            offset_to_ground: Shift each frame to ground clearance.

        Returns:
            Typed robot motion.
        """
        if not motion.frames:
            raise ValueError("human motion must contain at least one frame")
        qpos = np.stack(
            [
                self.retarget_frame(
                    frame,
                    human_height=motion.height,
                    offset_to_ground=offset_to_ground,
                )
                for frame in motion.frames
            ]
        )
        qpos[:, :2] -= qpos[0, :2]
        return RobotMotion(
            robot=self.robot.identifier,
            source_format=motion.source_format,
            profile=self.profile.identifier,
            fps=motion.fps,
            root_positions=qpos[:, :3],
            root_quaternions=qpos[:, 3:7],
            joint_positions=qpos[:, 7:],
            joint_names=self._joint_names(),
            source_identifier=motion.source_identifier,
            source_fps=motion.source_fps,
        )
