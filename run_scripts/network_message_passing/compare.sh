#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIG_PATH="${1:-${ROOT_DIR}/data/network_message_passing/example_network.json}"
STEPS="${2:-180}"
OUTPUT_JSON="${3:-${ROOT_DIR}/logs/network_message_passing/compare.json}"
PLOT_PATH="${4:-${ROOT_DIR}/logs/network_message_passing/compare.png}"

bash "${ROOT_DIR}/run_scripts/network_message_passing/run.sh" \
  both \
  "${CONFIG_PATH}" \
  "${STEPS}" \
  "${OUTPUT_JSON}" \
  "${PLOT_PATH}"

