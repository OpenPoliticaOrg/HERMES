#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VIDEO_SOURCE="${1:-0}"
ECOLOGICAL_CONTEXT="${2:-salon}"
QUESTION="${3:-what is the activity in the video?}"
SEQUENCE_ID="${4:-cam0}"
OUTPUT_JSONL="${5:-${ROOT_DIR}/logs/context_markov_live_viz.jsonl}"
CHECKPOINT_PATH="${6:-}"
CONTEXT_OPTIONS="${7:-}"
MARKOV_ORDER="${8:-0}"
WINDOW_SIZE="${9:--1}"
TE_TARGET_ORDER="${10:-0}"
TE_SOURCE_ORDER="${11:-0}"

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

if [[ -n "${CONTEXT_OPTIONS}" ]]; then
  CMD+=(--context-options "${CONTEXT_OPTIONS}")
fi

if [[ "${MARKOV_ORDER}" -ge 1 ]]; then
  CMD+=(--markov-order-override "${MARKOV_ORDER}")
fi

if [[ "${WINDOW_SIZE}" -ge 0 ]]; then
  CMD+=(--markov-window-size-override "${WINDOW_SIZE}")
fi

if [[ "${TE_TARGET_ORDER}" -ge 1 || "${TE_SOURCE_ORDER}" -ge 1 ]]; then
  CMD+=(--te-mode-override symbolic_matrix)
fi

if [[ "${TE_TARGET_ORDER}" -ge 1 ]]; then
  CMD+=(--te-target-order-override "${TE_TARGET_ORDER}")
fi

if [[ "${TE_SOURCE_ORDER}" -ge 1 ]]; then
  CMD+=(--te-source-order-override "${TE_SOURCE_ORDER}")
fi

"${CMD[@]}"
