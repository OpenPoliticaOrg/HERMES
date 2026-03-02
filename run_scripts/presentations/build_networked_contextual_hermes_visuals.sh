#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SINGLE_JSON="${1:-${ROOT_DIR}/logs/network_message_passing/surveillance_single.json}"
SUMMARY_JSON="${2:-${ROOT_DIR}/logs/network_message_passing/monte_carlo_summary.json}"
OUT_DIR="${3:-${ROOT_DIR}/docs/assets/networked_contextual_hermes}"
N_RUNS="${4:-12}"
STEPS="${5:-320}"

# Set REBUILD_DATA=1 to force rerunning simulator outputs.
REBUILD_DATA="${REBUILD_DATA:-0}"

mkdir -p "${ROOT_DIR}/logs/network_message_passing"

if [[ "${REBUILD_DATA}" == "1" || ! -f "${SINGLE_JSON}" ]]; then
  bash "${ROOT_DIR}/run_scripts/network_message_passing/surveillance_test.sh" \
    both "${STEPS}" "${SINGLE_JSON}" \
    "${ROOT_DIR}/logs/network_message_passing/surveillance_single.png"
fi

if [[ "${REBUILD_DATA}" == "1" || ! -f "${SUMMARY_JSON}" ]]; then
  bash "${ROOT_DIR}/run_scripts/network_message_passing/surveillance_monte_carlo.sh" \
    "${N_RUNS}" "${STEPS}" \
    "${ROOT_DIR}/logs/network_message_passing/monte_carlo" \
    "${SUMMARY_JSON}"
fi

python "${ROOT_DIR}/tools/build_networked_contextual_hermes_visuals.py" \
  --repo-root "${ROOT_DIR}" \
  --single-run-json "${SINGLE_JSON}" \
  --summary-json "${SUMMARY_JSON}" \
  --output-dir "${OUT_DIR}"

