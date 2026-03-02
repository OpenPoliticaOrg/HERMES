#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

POLICY="${1:-both}"  # min_cost_lp | backpressure | both
CONFIG_PATH="${2:-${ROOT_DIR}/data/network_message_passing/example_network.json}"
STEPS="${3:-180}"
OUTPUT_JSON="${4:-${ROOT_DIR}/logs/network_message_passing/results.json}"
PLOT_PATH="${5:-${ROOT_DIR}/logs/network_message_passing/dashboard.png}"

mkdir -p "${ROOT_DIR}/logs/network_message_passing"

python "${ROOT_DIR}/tools/network_message_passing_sim.py" \
  --repo-root "${ROOT_DIR}" \
  --config "${CONFIG_PATH}" \
  --policy "${POLICY}" \
  --steps "${STEPS}" \
  --output-json "${OUTPUT_JSON}" \
  --plot-path "${PLOT_PATH}"

