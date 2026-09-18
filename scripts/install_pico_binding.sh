#!/usr/bin/env bash

set -euo pipefail

readonly BINDING_REPOSITORY="https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git"
readonly BINDING_REVISION="75cb1130ac63e76d8f7e7788049be415e8be44f2"
readonly SERVICE_REPOSITORY="https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git"
readonly SERVICE_REVISION="85bac4dbc1fd5cef42c74a160d9c30aa3491f122"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer currently supports Linux only." >&2
  exit 1
fi

for command in git uv; do
  if ! command -v "${command}" >/dev/null; then
    echo "${command} is required." >&2
    exit 1
  fi
done

work_directory="$(mktemp -d)"
trap 'rm -rf "${work_directory}"' EXIT

binding_directory="${work_directory}/binding"
service_directory="${binding_directory}/tmp/XRoboToolkit-PC-Service"

git init --quiet "${binding_directory}"
git -C "${binding_directory}" remote add origin "${BINDING_REPOSITORY}"
git -C "${binding_directory}" fetch --quiet --depth 1 origin "${BINDING_REVISION}"
git -C "${binding_directory}" checkout --quiet --detach FETCH_HEAD

mkdir -p "${binding_directory}/tmp"
git init --quiet "${service_directory}"
git -C "${service_directory}" remote add origin "${SERVICE_REPOSITORY}"
git -C "${service_directory}" fetch --quiet --depth 1 origin "${SERVICE_REVISION}"
git -C "${service_directory}" checkout --quiet --detach FETCH_HEAD

sdk_directory="${service_directory}/RoboticsService/PXREARobotSDK"
uv pip install pybind11 cmake setuptools
uv run bash "${sdk_directory}/build.sh"

mkdir -p "${binding_directory}/include" "${binding_directory}/lib"
cp "${sdk_directory}/PXREARobotSDK.h" "${binding_directory}/include/"
cp -R "${sdk_directory}/nlohmann" "${binding_directory}/include/nlohmann"
cp "${sdk_directory}/build/libPXREARobotSDK.so" "${binding_directory}/lib/"

uv pip install --no-build-isolation "${binding_directory}"
uv run python -c "import xrobotoolkit_sdk"
echo "PICO Python binding installed successfully."
