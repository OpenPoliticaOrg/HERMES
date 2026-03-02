#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT_GLOB="${1:-${ROOT_DIR}/logs/network_message_passing/monte_carlo/run_*.json}"
OUTPUT_JSON="${2:-${ROOT_DIR}/logs/network_message_passing/monte_carlo_summary.json}"
RANKING_WEIGHTS="${3:-}"

CMD=(
  python "${ROOT_DIR}/tools/network_message_passing_kpi_report.py"
  --repo-root "${ROOT_DIR}"
  --input-glob "${INPUT_GLOB}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${RANKING_WEIGHTS}" ]]; then
  CMD+=(--ranking-weights "${RANKING_WEIGHTS}")
fi

"${CMD[@]}"
