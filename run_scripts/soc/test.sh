#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROOT_DIR}/tools/soc_readiness_smoke.py" \
  --repo-root "${ROOT_DIR}" \
  --print-json
