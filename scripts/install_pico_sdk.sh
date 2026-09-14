#!/usr/bin/env bash
# Install the XRoboToolkit Python SDK (xrobotoolkit_sdk) for PICO streaming.
#
# Requires the XRoboToolkit PC Service deb to be installed first (see the
# README PICO section): it ships the native SDK at /opt/apps/roboticsservice,
# so only the small Python binding is compiled here, against the exact
# library version the running service uses.
#
# Installs into the active Python environment if one is activated (uv .venv
# or conda), otherwise into the .venv of this GMR checkout (from `uv sync`).
#
# Usage, from the GMR repo root:
#   bash scripts/install_pico_sdk.sh
set -euo pipefail

DEB_SDK="/opt/apps/roboticsservice/SDK"

if [[ ! -f "$DEB_SDK/x64/libPXREARobotSDK.so" ]]; then
    echo "XRoboToolkit PC Service SDK not found at $DEB_SDK." >&2
    echo "Install the PC Service deb first (see the PICO section of the README)." >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
        echo "No Python environment is active and $ROOT/.venv does not exist." >&2
        echo "Run 'uv sync' first, or activate the environment to install into." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

echo "Cloning the Python binding..."
git clone --depth 1 https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
cd XRoboToolkit-PC-Service-Pybind

# The binding repo ships the headers; the deb provides the matching library.
mkdir -p lib
cp "$DEB_SDK/x64/libPXREARobotSDK.so" lib/
cp "$DEB_SDK/include/PXREARobotSDK.h" include/

echo "Building and installing the Python binding..."
pip install pybind11 cmake
pip uninstall -y xrobotoolkit_sdk || true
python setup.py install

echo "xrobotoolkit_sdk installed into $(python -c 'import sys; print(sys.prefix)')"
