"""Tests for the package import boundary."""

import subprocess
import sys


def test_core_api_does_not_import_heavy_subsystems() -> None:
    code = """
import sys
import general_motion_retargeting as gmr

heavy = {"imageio", "mjviser", "smplx", "torch", "viser"}
assert heavy.isdisjoint(sys.modules)
assert gmr.RetargetApplication
assert heavy.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
