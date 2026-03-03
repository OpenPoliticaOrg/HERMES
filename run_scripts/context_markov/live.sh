#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VIDEO_SOURCE="${1:-0}"
ECOLOGICAL_CONTEXT="${2:-salon}"
QUESTION="${3:-what is the activity in the video?}"
SEQUENCE_ID="${4:-cam0}"
OUTPUT_JSONL="${5:-${ROOT_DIR}/logs/context_markov_live.jsonl}"
CHECKPOINT_PATH="${6:-}"
ENTITY_OBS_BY_WINDOW="${7:-}"
ENTITY_MISSING_TOL="${8:-0}"
ENTITY_OBS_MODE="${9:-none}"

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
  --options
  run.classification_mode rank
  run.event_taxonomy_path "${ROOT_DIR}/data/taxonomy/example_event_taxonomy.json"
  run.observation_classifier_path "${ROOT_DIR}/data/taxonomy/example_observation_classifiers.json"
  run.markov_chain_path "${ROOT_DIR}/data/taxonomy/example_markov_chain.json"
  run.markov_context_field ecological_context
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi

if [[ -n "${ENTITY_OBS_BY_WINDOW}" ]]; then
  CMD+=(--entity-observations-by-window "${ENTITY_OBS_BY_WINDOW}")
fi

if [[ "${ENTITY_MISSING_TOL}" -ge 0 ]]; then
  CMD+=(--entity-missing-tolerance "${ENTITY_MISSING_TOL}")
fi

if [[ "${ENTITY_OBS_MODE}" != "none" ]]; then
  CMD+=(--entity-observation-mode "${ENTITY_OBS_MODE}")
fi

"${CMD[@]}"
