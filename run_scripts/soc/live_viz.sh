#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VIDEO_SOURCE="${1:-0}"
ECOLOGICAL_CONTEXT="${2:-warehouse_entry}"
QUESTION="${3:-what is the activity in the video?}"
SEQUENCE_ID="${4:-cam_soc0}"
OUTPUT_JSONL="${5:-${ROOT_DIR}/logs/soc_live_viz.jsonl}"
CHECKPOINT_PATH="${6:-}"
SOC_CONFIG="${7:-${ROOT_DIR}/data/soc/example_soc_runtime_config.json}"

mkdir -p "${ROOT_DIR}/logs"

CMD=(
  python "${ROOT_DIR}/stream_online.py"
  --cfg-path "${ROOT_DIR}/lavis/projects/hermes/cls_coin.yaml"
  --video-source "${VIDEO_SOURCE}"
  --question "${QUESTION}"
  --sequence-id "${SEQUENCE_ID}"
  --ecological-context "${ECOLOGICAL_CONTEXT}"
  --context-field ecological_context
  --chunk-seconds 3
  --stride-seconds 1
  --output-jsonl "${OUTPUT_JSONL}"
  --debug-markov
  --visualize matplotlib
  --interactive-context
  --soc-enable
  --soc-config "${SOC_CONFIG}"
  --soc-print-routing
  --options
  run.classification_mode rank
  run.event_taxonomy_path "${ROOT_DIR}/data/taxonomy/example_event_taxonomy.json"
  run.observation_classifier_path "${ROOT_DIR}/data/taxonomy/example_observation_classifiers.json"
  run.markov_chain_path "${ROOT_DIR}/data/taxonomy/example_markov_chain.json"
  run.markov_context_field ecological_context
  run.markov_debug True
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi

"${CMD[@]}"
