#!/usr/bin/env bash

set -euo pipefail

readonly PARSER_VERSION="0.1.6"
readonly PARSER_REVISION="ac89d303bfb9b14ae02dbe424a9fd3dd9872eb3e"
readonly PARSER_BASE_URL="https://raw.githubusercontent.com/jiminghe/xsens_mvn_robot_python"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The Xsens parser only provides Linux wheels." >&2
  exit 1
fi

python_tag="$(
  uv run python -c \
    'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")'
)"
case "${python_tag}" in
  cp310 | cp311 | cp312 | cp313) ;;
  *)
    echo "No Xsens parser wheel is available for ${python_tag}." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64)
    architecture="x86_64"
    ;;
  aarch64 | arm64)
    architecture="aarch64"
    ;;
  *)
    echo "No Xsens parser wheel is available for $(uname -m)." >&2
    exit 1
    ;;
esac

wheel="xsens_mvn_robot-${PARSER_VERSION}-${python_tag}-none-linux_${architecture}.whl"
url="${PARSER_BASE_URL}/${PARSER_REVISION}/${wheel}"

echo "Installing ${wheel}"
uv pip install "${url}"
uv run python -c "from xsens_mvn_robot import XsensWrapper"
echo "Xsens parser installed successfully."
