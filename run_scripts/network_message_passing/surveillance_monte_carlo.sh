#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

N_RUNS="${1:-20}"
STEPS="${2:-320}"
OUT_DIR="${3:-${ROOT_DIR}/logs/network_message_passing/monte_carlo}"
SUMMARY_JSON="${4:-${ROOT_DIR}/logs/network_message_passing/monte_carlo_summary.json}"

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/run_*.json

for ((s=1; s<=N_RUNS; s++)); do
  OUT_FILE="${OUT_DIR}/run_${s}.json"
  echo "Running seed ${s}/${N_RUNS} -> ${OUT_FILE}"
  python "${ROOT_DIR}/tools/network_message_passing_sim.py" \
    --repo-root "${ROOT_DIR}" \
    --config "${ROOT_DIR}/data/network_message_passing/surveillance_mesh.json" \
    --policy both \
    --steps "${STEPS}" \
    --seed "${s}" \
    --output-json "${OUT_FILE}" >/dev/null
done

python "${ROOT_DIR}/tools/network_message_passing_kpi_report.py" \
  --repo-root "${ROOT_DIR}" \
  --input-glob "${OUT_DIR}/run_*.json" \
  --output-json "${SUMMARY_JSON}"
