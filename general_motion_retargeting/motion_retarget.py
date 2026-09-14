import json
from dataclasses import dataclass, field

import mink
import mujoco as mj
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R

from .config_validation import validate_human_frame, validate_ik_config
from .model_loader import load_robot_model
from .params import IK_CONFIG_DICT, ROBOT_XML_DICT

# One frame of human motion: body name -> [position (3,), wxyz quaternion (4,)].
HumanFrame = dict[str, list[np.ndarray]]


@dataclass
class _IKStage:
    """Tasks and per-body offsets built from one ik_match_table."""

    tasks: list[mink.FrameTask] = field(default_factory=list)
    task_by_human_body: dict[str, mink.FrameTask] = field(default_factory=dict)
    pos_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    rot_offsets: dict[str, R] = field(default_factory=dict)


class GeneralMotionRetargeting:
    """Retargets human motion frames to a robot via two-stage differential IK.

    Each frame of human data (body name -> position and wxyz quaternion) is
    scaled, offset, and fed as targets to two prioritized groups of mink
    frame tasks, defined by the ik_match_table1/2 entries of the IK config.
    """

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float | None = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = True,
        use_velocity_limit: bool = False,
    ) -> None:
        """Load the robot model and IK config and build the IK tasks.

        Args:
            src_human: Source motion format key in IK_CONFIG_DICT (e.g. "smplx").
            tgt_robot: Target robot key in ROBOT_XML_DICT (e.g. "unitree_g1").
            actual_human_height: Height of the motion subject in meters; scale
                factors are adjusted relative to the config's height assumption.
                None keeps the config's scale table as-is.
            solver: QP solver name passed to mink.solve_ik.
            damping: Levenberg-Marquardt damping passed to mink.solve_ik.
            verbose: Print the robot's DoF/body/motor tables and config path.
            use_velocity_limit: Cap joint velocities at 3*pi rad/s per step.
        """
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)

        config_path = IK_CONFIG_DICT[src_human][tgt_robot]
        with open(config_path) as f:
            ik_config = json.load(f)

        # Tracking sites declared in the config are injected into the model,
        # so configs can target frames the vendor XML does not define.
        self.model = load_robot_model(self.xml_file, ik_config.get("tracking_sites"))

        self.robot_dof_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i]): i
            for i in range(self.model.nv)
        }
        self.robot_body_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i): i
            for i in range(self.model.nbody)
        }
        self.robot_motor_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }
        self.robot_site_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SITE, i): i
            for i in range(self.model.nsite)
        }
        if verbose:
            print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
            for name, i in self.robot_dof_names.items():
                print(f"DoF {i}: {name}")
            print("[GMR] Robot Body names and their IDs:")
            for name, i in self.robot_body_names.items():
                print(f"Body ID {i}: {name}")
            print("[GMR] Robot Motor (Actuator) names and their IDs:")
            for name, i in self.robot_motor_names.items():
                print(f"Motor ID {i}: {name}")

        validate_ik_config(ik_config, self.model, str(config_path))
        if verbose:
            print("Use IK config: ", config_path)

        # Scale factors are calibrated for the config's height assumption;
        # rescale them for the actual subject height when it is known.
        ratio = 1.0
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        self.human_scale_table: dict[str, float] = {
            key: value * ratio for key, value in ik_config["human_scale_table"].items()
        }

        self.ik_match_table1: dict[str, list] = ik_config["ik_match_table1"]
        self.ik_match_table2: dict[str, list] = ik_config["ik_match_table2"]
        self.human_root_name: str = ik_config["human_root_name"]
        self.robot_root_name: str = ik_config["robot_root_name"]
        self.use_ik_match_table1: bool = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2: bool = ik_config["use_ik_match_table2"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10
        self.solver = solver
        self.damping = damping

        self.ik_limits: list = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            velocity_limits = {name: 3 * np.pi for name in self.robot_motor_names}
            self.ik_limits.append(mink.VelocityLimit(self.model, velocity_limits))

        self.setup_retarget_configuration()

        self.required_human_bodies = {self.human_root_name}
        if self.use_ik_match_table1:
            self.required_human_bodies |= set(self.stage1.task_by_human_body)
        if self.use_ik_match_table2:
            self.required_human_bodies |= set(self.stage2.task_by_human_body)

        self.ground_offset = 0.0

    def setup_retarget_configuration(self) -> None:
        """Build the mink configuration and the two task stages."""
        self.configuration = mink.Configuration(self.model)
        self.stage1 = self._build_stage(self.ik_match_table1)
        self.stage2 = self._build_stage(self.ik_match_table2)

    def _build_stage(self, ik_match_table: dict[str, list]) -> _IKStage:
        """Create frame tasks and offsets for every weighted table entry."""
        stage = _IKStage()
        for frame_name, entry in ik_match_table.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight == 0 and rot_weight == 0:
                continue
            # Bodies take precedence so that a site sharing a body's name
            # (e.g. g1's imu_in_torso) keeps the pre-existing behavior.
            frame_type = "body" if frame_name in self.robot_body_names else "site"
            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type=frame_type,
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            stage.tasks.append(task)
            stage.task_by_human_body[body_name] = task
            stage.pos_offsets[body_name] = np.array(pos_offset) - self.ground
            stage.rot_offsets[body_name] = R.from_quat(rot_offset, scalar_first=True)
        return stage

    # Aliases kept for external users of the pre-refactor attribute names.
    @property
    def tasks1(self) -> list:
        return self.stage1.tasks

    @property
    def tasks2(self) -> list:
        return self.stage2.tasks

    @property
    def human_body_to_task1(self) -> dict:
        return self.stage1.task_by_human_body

    @property
    def human_body_to_task2(self) -> dict:
        return self.stage2.task_by_human_body

    @property
    def pos_offsets1(self) -> dict:
        return self.stage1.pos_offsets

    @property
    def rot_offsets1(self) -> dict:
        return self.stage1.rot_offsets

    @property
    def pos_offsets2(self) -> dict:
        return self.stage2.pos_offsets

    @property
    def rot_offsets2(self) -> dict:
        return self.stage2.rot_offsets

    def _active_stages(self) -> list[_IKStage]:
        stages = []
        if self.use_ik_match_table1:
            stages.append(self.stage1)
        if self.use_ik_match_table2:
            stages.append(self.stage2)
        return stages

    def update_targets(
        self, human_data: HumanFrame, offset_to_ground: bool = False
    ) -> None:
        """Scale and offset one human frame and set it as the task targets.

        Args:
            human_data: Frame mapping body names to [position, wxyz quaternion].
            offset_to_ground: Shift the whole frame so the lowest foot rests
                slightly above the ground.

        Raises:
            ValueError: If the frame is missing bodies the IK config tracks.
        """
        validate_human_frame(human_data, self.required_human_bodies)
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(
            human_data, self.human_root_name, self.human_scale_table
        )
        human_data = self.offset_human_data(
            human_data, self.stage1.pos_offsets, self.stage1.rot_offsets
        )
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        for stage in self._active_stages():
            for body_name, task in stage.task_by_human_body.items():
                pos, rot = human_data[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

    def retarget(
        self, human_data: HumanFrame, offset_to_ground: bool = False
    ) -> np.ndarray:
        """Retarget one human frame and return the robot qpos.

        Args:
            human_data: Frame mapping body names to [position, wxyz quaternion].
            offset_to_ground: See update_targets.

        Returns:
            Copy of the resulting qpos: floating base pose followed by joints.
        """
        self.update_targets(human_data, offset_to_ground)
        for stage in self._active_stages():
            self._solve_stage(stage.tasks)
        return self.configuration.data.qpos.copy()

    def _solve_stage(self, tasks: list) -> None:
        """Iterate differential IK on one task group until error stagnates."""
        dt = self.configuration.model.opt.timestep
        curr_error = self._task_error(tasks)
        velocity = mink.solve_ik(
            self.configuration,
            tasks,
            dt,
            self.solver,
            self.damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(velocity, dt)
        next_error = self._task_error(tasks)
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < self.max_iter:
            curr_error = next_error
            velocity = mink.solve_ik(
                self.configuration,
                tasks,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(velocity, dt)
            next_error = self._task_error(tasks)
            num_iter += 1

    def _task_error(self, tasks: list) -> float:
        return np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in tasks])
        )

    def error1(self) -> float:
        return self._task_error(self.stage1.tasks)

    def error2(self) -> float:
        return self._task_error(self.stage2.tasks)

    def to_numpy(self, human_data: HumanFrame) -> HumanFrame:
        """Convert every position and quaternion in the frame to numpy arrays."""
        for body_name in human_data.keys():
            human_data[body_name] = [
                np.asarray(human_data[body_name][0]),
                np.asarray(human_data[body_name][1]),
            ]
        return human_data

    def scale_human_data(
        self,
        human_data: HumanFrame,
        human_root_name: str,
        human_scale_table: dict[str, float],
    ) -> HumanFrame:
        """Scale body positions around the root, per the scale table.

        Bodies absent from the scale table are dropped from the result.
        """
        root_pos, root_quat = human_data[human_root_name]
        scaled_root_pos = human_scale_table[human_root_name] * root_pos

        scaled = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name, (pos, quat) in human_data.items():
            if body_name not in human_scale_table or body_name == human_root_name:
                continue
            local_pos = (pos - root_pos) * human_scale_table[body_name]
            scaled[body_name] = (local_pos + scaled_root_pos, quat)
        return scaled

    def offset_human_data(
        self,
        human_data: HumanFrame,
        pos_offsets: dict[str, np.ndarray],
        rot_offsets: dict[str, R],
    ) -> HumanFrame:
        """Apply per-body rotation offsets and local-frame position offsets."""
        offset = {}
        for body_name, (pos, quat) in human_data.items():
            rotation = R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]
            updated_quat = rotation.as_quat(scalar_first=True)
            global_pos_offset = rotation.apply(pos_offsets[body_name])
            offset[body_name] = [pos + global_pos_offset, updated_quat]
        return offset

    def offset_human_data_to_ground(self, human_data: HumanFrame) -> HumanFrame:
        """Shift the frame so the lowest foot sits 0.1 m above the ground."""
        ground_offset = 0.1
        lowest_pos = np.inf
        for body_name, (pos, _quat) in human_data.items():
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            lowest_pos = min(lowest_pos, pos[2])
        shift = np.array([0, 0, lowest_pos]) - np.array([0, 0, ground_offset])
        return {
            body_name: [pos - shift, quat]
            for body_name, (pos, quat) in human_data.items()
        }

    def set_ground_offset(self, ground_offset: float) -> None:
        """Set a constant downward shift applied to every incoming frame."""
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data: HumanFrame) -> HumanFrame:
        """Subtract the configured ground offset from every body position."""
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
