#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

POLICY="${1:-both}"  # min_cost_lp | backpressure | both
STEPS="${2:-320}"
OUTPUT_JSON="${3:-${ROOT_DIR}/logs/network_message_passing/surveillance_single.json}"
PLOT_PATH="${4:-${ROOT_DIR}/logs/network_message_passing/surveillance_single.png}"

mkdir -p "${ROOT_DIR}/logs/network_message_passing"

python "${ROOT_DIR}/tools/network_message_passing_sim.py" \
  --repo-root "${ROOT_DIR}" \
  --config "${ROOT_DIR}/data/network_message_passing/surveillance_mesh.json" \
  --policy "${POLICY}" \
  --steps "${STEPS}" \
  --output-json "${OUTPUT_JSON}" \
  --plot-path "${PLOT_PATH}"

