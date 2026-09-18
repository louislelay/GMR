"""Tests for SMPL-X body model extension detection."""

import pytest

from general_motion_retargeting.utils.smpl import (
    _normalize_gender,
    detect_body_model_ext,
)


def make_body_model(tmp_path, name):
    model_dir = tmp_path / "smplx"
    model_dir.mkdir(exist_ok=True)
    (model_dir / name).touch()


def test_detects_npz(tmp_path):
    make_body_model(tmp_path, "SMPLX_NEUTRAL.npz")
    assert detect_body_model_ext(str(tmp_path), "neutral") == "npz"


def test_detects_pkl(tmp_path):
    make_body_model(tmp_path, "SMPLX_MALE.pkl")
    assert detect_body_model_ext(str(tmp_path), "male") == "pkl"


def test_prefers_npz_when_both_exist(tmp_path):
    make_body_model(tmp_path, "SMPLX_NEUTRAL.npz")
    make_body_model(tmp_path, "SMPLX_NEUTRAL.pkl")
    assert detect_body_model_ext(str(tmp_path), "neutral") == "npz"


def test_missing_model_raises_with_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="SMPLX_FEMALE.npz"):
        detect_body_model_ext(str(tmp_path), "female")


def test_normalizes_byte_gender():
    assert _normalize_gender(b"female") == "female"


def test_rejects_unknown_gender():
    with pytest.raises(ValueError, match="Unsupported SMPL-X gender"):
        _normalize_gender("unknown")
