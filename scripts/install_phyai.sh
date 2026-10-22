#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="${UV_BIN:-/root/.local/bin/uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/uv-cache}"
export UV_LOCK_TIMEOUT=1200
if [[ ! -d third_party/phyai/.git ]]; then
  git clone https://github.com/mingti-org/phyai.git third_party/phyai
  git -C third_party/phyai checkout f983a8f84684aa7db4dfac7184b7b5c787635978
fi
cd third_party/phyai
"$uv_bin" sync --locked --no-dev
