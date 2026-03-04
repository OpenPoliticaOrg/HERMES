#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

STEPS="${1:-24}"
CAMERAS="${2:-cam_a01,cam_a02,cam_a03}"
OUTPUT_JSON="${3:-${ROOT_DIR}/logs/soc_coordination/summary.json}"
SOC_CONFIG="${4:-${ROOT_DIR}/data/soc/example_soc_runtime_config.json}"

python "${ROOT_DIR}/tools/soc_coordination_sim.py" \
  --repo-root "${ROOT_DIR}" \
  --soc-config "${SOC_CONFIG}" \
  --steps "${STEPS}" \
  --cameras "${CAMERAS}" \
  --output-json "${OUTPUT_JSON}" \
  --print-json
