#!/bin/bash
# Install the lemonade-burn backend adapter into a Lemonade v11.9.0 source tree and
# register it in CMakeLists.txt (LEMON_BACKENDS). Rebuild lemond afterwards.
# Usage: install_into_lemonade.sh /path/to/lemonade/src
set -euo pipefail
SRC="${1:?lemonade source dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$SRC/src/cpp/include/lemon/backends/burn" "$SRC/src/cpp/server/backends/burn"
cp "$HERE"/src/cpp/include/lemon/backends/burn/*.h "$SRC/src/cpp/include/lemon/backends/burn/"
cp "$HERE"/src/cpp/server/backends/burn/*.cpp "$SRC/src/cpp/server/backends/burn/"
if ! grep -q '"lemonade-burn|burn"' "$SRC/CMakeLists.txt"; then
  sed -i 's/^    "cake|cake"$/    "cake|cake"\n    "lemonade-burn|burn"/' "$SRC/CMakeLists.txt"
fi
grep -n '"lemonade-burn|burn"' "$SRC/CMakeLists.txt"
