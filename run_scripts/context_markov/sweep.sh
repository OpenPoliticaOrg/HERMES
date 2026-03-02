#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MARKOV_CFG="${1:-${ROOT_DIR}/data/taxonomy/example_markov_chain.json}"
ORDERS="${2:-1,2,3}"
WINDOWS="${3:-0,6,12}"
TE_TARGET_ORDERS="${4:-1,2}"
TE_SOURCE_ORDERS="${5:-1,2}"
STEPS_PER_PHASE="${6:-25}"
OUTPUT_JSON="${7:-${ROOT_DIR}/logs/context_markov_sweep.json}"

mkdir -p "${ROOT_DIR}/logs"

python "${ROOT_DIR}/tools/context_markov_sweep.py" \
  --repo-root "${ROOT_DIR}" \
  --markov "${MARKOV_CFG}" \
  --orders "${ORDERS}" \
  --window-sizes "${WINDOWS}" \
  --te-mode symbolic_matrix \
  --te-target-orders "${TE_TARGET_ORDERS}" \
  --te-source-orders "${TE_SOURCE_ORDERS}" \
  --steps-per-phase "${STEPS_PER_PHASE}" \
  --output-json "${OUTPUT_JSON}"

