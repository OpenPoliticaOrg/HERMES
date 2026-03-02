#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CHECKPOINT_PATH="${1:-}"
VIDEO_SOURCE="${2:-}"

CMD=(
  python "${ROOT_DIR}/tools/context_markov_doctor.py"
  --repo-root "${ROOT_DIR}"
  --run-smoke
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ -n "${VIDEO_SOURCE}" ]]; then
  CMD+=(--video-source "${VIDEO_SOURCE}")
fi

"${CMD[@]}"
