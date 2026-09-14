"""Tests for tracking-site injection into robot models."""

import json

import mujoco as mj
import numpy as np
import pytest
from test_retarget_regression import DATA_DIR, load_synthetic_motion, retarget_motion

from general_motion_retargeting import IK_CONFIG_DICT, GeneralMotionRetargeting
from general_motion_retargeting.model_loader import load_robot_model

G1_XML = "assets/unitree_g1/g1_mocap_29dof.xml"

# The toe trackers are dummy bodies hand-added to the shipped g1 XML; these
# sites reproduce them on the stock parent bodies at the same offsets.
TOE_SITES = {
    "left_toe_site": {"body": "left_ankle_roll_link", "pos": [0.1, 0.0, -0.02]},
    "right_toe_site": {"body": "right_ankle_roll_link", "pos": [0.1, 0.0, -0.02]},
}


def test_loader_injects_sites():
    model = load_robot_model(G1_XML, TOE_SITES)
    for name, entry in TOE_SITES.items():
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        assert site_id >= 0
        np.testing.assert_allclose(model.site_pos[site_id], entry["pos"])


def test_loader_rejects_unknown_parent():
    with pytest.raises(ValueError, match="parent body 'nope' does not exist"):
        load_robot_model(G1_XML, {"x": {"body": "nope"}})


def test_loader_without_sites_matches_plain_load():
    plain = mj.MjModel.from_xml_path(G1_XML)
    loaded = load_robot_model(G1_XML)
    assert loaded.nbody == plain.nbody
    assert loaded.nsite == plain.nsite
    assert loaded.nq == plain.nq


@pytest.fixture
def site_config_path(tmp_path, monkeypatch):
    """A g1 config whose toe trackers use injected sites, not dummy bodies."""
    with open(IK_CONFIG_DICT["smplx"]["unitree_g1"]) as f:
        config = json.load(f)
    config["tracking_sites"] = TOE_SITES
    for table in ("ik_match_table1", "ik_match_table2"):
        for side in ("left", "right"):
            config[table][f"{side}_toe_site"] = config[table].pop(f"{side}_toe_link")
    path = tmp_path / "smplx_to_g1_sites.json"
    path.write_text(json.dumps(config))
    monkeypatch.setitem(IK_CONFIG_DICT["smplx"], "unitree_g1", path)
    return path


def test_site_tasks_match_golden(site_config_path):
    """Tracking via injected sites reproduces the dummy-body golden exactly.

    A site injected on the stock parent body at the dummy body's offset defines
    the same frame, so the retargeted qpos must match the committed golden.
    """
    frames = load_synthetic_motion()
    qpos = retarget_motion("unitree_g1", frames)
    golden = np.load(DATA_DIR / "golden_qpos_unitree_g1.npy")
    np.testing.assert_allclose(qpos, golden, rtol=0.0, atol=1e-4)


def test_site_frames_pass_validation(site_config_path):
    retargeter = GeneralMotionRetargeting("smplx", "unitree_g1", verbose=False)
    assert "left_toe_site" in retargeter.robot_site_names
    assert "left_toe_link" in retargeter.robot_body_names
