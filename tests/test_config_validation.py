"""Tests for IK config and human-frame validation."""

import json

import mujoco as mj
import pytest
from generate_synthetic_motion import TPOSE_POSITIONS, TPOSE_QUAT

from general_motion_retargeting import (
    IK_CONFIG_DICT,
    ROBOT_XML_DICT,
    GeneralMotionRetargeting,
)
from general_motion_retargeting.config_validation import (
    validate_human_frame,
    validate_ik_config,
)

ALL_CONFIGS = [
    (src, robot) for src, robots in IK_CONFIG_DICT.items() for robot in robots
]

_MODEL_CACHE = {}


def load_model(robot):
    xml = ROBOT_XML_DICT.get(robot)
    if xml is None or not xml.exists():
        pytest.skip(f"robot assets for {robot} not available")
    if robot not in _MODEL_CACHE:
        _MODEL_CACHE[robot] = mj.MjModel.from_xml_path(str(xml))
    return _MODEL_CACHE[robot]


def load_config(src, robot):
    with open(IK_CONFIG_DICT[src][robot]) as f:
        return json.load(f)


@pytest.mark.parametrize("src,robot", ALL_CONFIGS)
def test_shipped_configs_are_valid(src, robot):
    model = load_model(robot)
    config = load_config(src, robot)
    validate_ik_config(config, model, f"{src}->{robot}")


def expect_error(config, model, match):
    with pytest.raises(ValueError, match=match):
        validate_ik_config(config, model, "test-config")


def test_missing_key():
    model = load_model("unitree_g1")
    config = load_config("smplx", "unitree_g1")
    del config["human_root_name"]
    expect_error(config, model, "missing keys: human_root_name")


def test_unknown_robot_body_suggests_close_match():
    model = load_model("unitree_g1")
    config = load_config("smplx", "unitree_g1")
    config["ik_match_table1"]["pelvis_typo"] = config["ik_match_table1"].pop("pelvis")
    expect_error(config, model, r"'pelvis_typo'.*did you mean 'pelvis'")


def test_negative_weight():
    model = load_model("unitree_g1")
    config = load_config("smplx", "unitree_g1")
    config["ik_match_table1"]["pelvis"][1] = -5.0
    expect_error(config, model, "invalid pos_weight: -5.0")


def test_non_unit_rot_offset():
    model = load_model("unitree_g1")
    config = load_config("smplx", "unitree_g1")
    config["ik_match_table1"]["pelvis"][4] = [0.0, 0.0, 0.0, 0.0]
    expect_error(config, model, "unit wxyz quaternion")


def test_tracked_body_missing_from_scale_table():
    model = load_model("unitree_g1")
    config = load_config("smplx", "unitree_g1")
    del config["human_scale_table"]["left_wrist"]
    expect_error(config, model, "left_wrist")


def test_human_frame_missing_bodies():
    with pytest.raises(ValueError, match="left_wrist, right_wrist"):
        validate_human_frame(
            {"pelvis": None, "left_foot": None},
            {"pelvis", "left_foot", "left_wrist", "right_wrist"},
        )


def test_retarget_rejects_incomplete_frame():
    retargeter = GeneralMotionRetargeting("smplx", "unitree_g1", verbose=False)
    frame = {
        body: [list(position), list(TPOSE_QUAT)]
        for body, position in TPOSE_POSITIONS.items()
    }
    del frame["left_wrist"]
    with pytest.raises(ValueError, match="left_wrist"):
        retargeter.retarget(frame)
