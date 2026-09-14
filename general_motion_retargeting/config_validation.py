"""Actionable validation for IK configs and incoming human motion frames.

Raises ValueError with messages that name the offending config entry or body
instead of failing later with a bare KeyError inside the retargeting loop.
"""

import difflib
import math

import mujoco as mj

_REQUIRED_KEYS = (
    "robot_root_name",
    "human_root_name",
    "ground_height",
    "human_height_assumption",
    "use_ik_match_table1",
    "use_ik_match_table2",
    "human_scale_table",
    "ik_match_table1",
    "ik_match_table2",
)


def _model_body_names(model):
    names = []
    for body_id in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if name is not None:
            names.append(name)
    return names


def _suggest(name, candidates):
    matches = difflib.get_close_matches(name, candidates, n=1)
    if matches:
        return f" (did you mean {matches[0]!r}?)"
    return ""


def _check_finite(values, what, where):
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"{where}: {what} contains non-finite values: {values}")


def _validate_entry(frame_name, entry, model_bodies, scale_table, where):
    if not isinstance(entry, (list, tuple)) or len(entry) != 5:
        raise ValueError(
            f"{where}: entry {frame_name!r} must be "
            "[human_body, pos_weight, rot_weight, pos_offset, rot_offset]"
        )
    human_body, pos_weight, rot_weight, pos_offset, rot_offset = entry

    if frame_name not in model_bodies:
        raise ValueError(
            f"{where}: entry {frame_name!r} references a body that does not "
            f"exist in the robot model{_suggest(frame_name, model_bodies)}"
        )

    for label, weight in (("pos_weight", pos_weight), ("rot_weight", rot_weight)):
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{where}: entry {frame_name!r} has invalid {label}: {weight}"
            )

    if len(pos_offset) != 3:
        raise ValueError(f"{where}: entry {frame_name!r} pos_offset must have 3 values")
    _check_finite(pos_offset, f"entry {frame_name!r} pos_offset", where)

    if len(rot_offset) != 4:
        raise ValueError(
            f"{where}: entry {frame_name!r} rot_offset must have 4 values (wxyz)"
        )
    _check_finite(rot_offset, f"entry {frame_name!r} rot_offset", where)
    norm = math.sqrt(sum(float(v) ** 2 for v in rot_offset))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(
            f"{where}: entry {frame_name!r} rot_offset must be a unit wxyz "
            f"quaternion (norm is {norm:.4f})"
        )

    is_active = float(pos_weight) != 0 or float(rot_weight) != 0
    if is_active and human_body not in scale_table:
        raise ValueError(
            f"{where}: entry {frame_name!r} tracks human body {human_body!r}, "
            "which is missing from human_scale_table"
            f"{_suggest(human_body, list(scale_table))}"
        )


def validate_ik_config(ik_config, model, config_path):
    """Validate a loaded IK config against the loaded robot model.

    Args:
        ik_config: Parsed JSON dict, before any in-place scaling.
        model: Loaded MjModel of the target robot.
        config_path: Config path, used in error messages.

    Raises:
        ValueError: On the first problem found, with an actionable message.
    """
    where = f"IK config {config_path}"

    missing = [key for key in _REQUIRED_KEYS if key not in ik_config]
    if missing:
        raise ValueError(f"{where} is missing keys: {', '.join(missing)}")

    height = float(ik_config["human_height_assumption"])
    if not math.isfinite(height) or height <= 0:
        raise ValueError(f"{where}: human_height_assumption must be positive")

    scale_table = ik_config["human_scale_table"]
    for body_name, scale in scale_table.items():
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"{where}: human_scale_table[{body_name!r}] must be a positive "
                f"finite number, got {scale}"
            )
    human_root = ik_config["human_root_name"]
    if human_root not in scale_table:
        raise ValueError(
            f"{where}: human root {human_root!r} is missing from human_scale_table"
        )

    if not ik_config["use_ik_match_table1"] and not ik_config["use_ik_match_table2"]:
        raise ValueError(f"{where}: at least one IK match table must be enabled")

    # Both tables are validated regardless of the use flags: the retargeter
    # builds tasks and offsets from both tables unconditionally.
    model_bodies = _model_body_names(model)
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for frame_name, entry in ik_config[table_name].items():
            _validate_entry(
                frame_name,
                entry,
                model_bodies,
                scale_table,
                f"{where} [{table_name}]",
            )

    # Every scaled human body is offset via its ik_match_table1 entry before
    # target update; a scale-table body without a weighted table1 entry would
    # raise a KeyError at runtime instead.
    offset_bodies = {
        entry[0]
        for entry in ik_config["ik_match_table1"].values()
        if float(entry[1]) != 0 or float(entry[2]) != 0
    }
    uncovered = sorted(set(scale_table) - offset_bodies)
    if uncovered:
        raise ValueError(
            f"{where}: human_scale_table bodies have no weighted ik_match_table1 "
            f"entry (needed for offsets): {', '.join(uncovered)}"
        )


def validate_human_frame(human_data, required_bodies):
    """Check that one human frame contains every body the IK config tracks.

    Args:
        human_data: Frame dict mapping human body names to (position, quat).
        required_bodies: Body names referenced by the enabled match tables.

    Raises:
        ValueError: Naming the missing bodies, instead of a later KeyError.
    """
    missing = sorted(set(required_bodies) - set(human_data))
    if missing:
        available = ", ".join(sorted(human_data))
        raise ValueError(
            "human frame is missing bodies required by the IK config: "
            f"{', '.join(missing)} (frame contains: {available})"
        )
