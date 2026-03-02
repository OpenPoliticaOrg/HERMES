#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLIDES_MD="${1:-${ROOT_DIR}/docs/presentation_networked_contextual_hermes.md}"
OUTPUT_HTML="${2:-${ROOT_DIR}/docs/presentation_networked_contextual_hermes.html}"
OUTPUT_PDF="${3:-${ROOT_DIR}/docs/presentation_networked_contextual_hermes.pdf}"
GENERATE_VISUALS="${GENERATE_VISUALS:-1}"

if [[ "${GENERATE_VISUALS}" == "1" ]]; then
  bash "${ROOT_DIR}/run_scripts/presentations/build_networked_contextual_hermes_visuals.sh"
fi

MARP_CMD=()
if command -v marp >/dev/null 2>&1; then
  MARP_CMD=("marp")
elif command -v npx >/dev/null 2>&1; then
  MARP_CMD=("npx" "--yes" "@marp-team/marp-cli")
else
  echo "Neither marp nor npx is available."
  echo "Install Node.js/npm or install marp globally:"
  echo "  npm install -g @marp-team/marp-cli"
  exit 1
fi

"${MARP_CMD[@]}" "${SLIDES_MD}" --allow-local-files --html -o "${OUTPUT_HTML}"
"${MARP_CMD[@]}" "${SLIDES_MD}" --allow-local-files --pdf -o "${OUTPUT_PDF}"

echo "Built web slides: ${OUTPUT_HTML}"
echo "Built static slides PDF: ${OUTPUT_PDF}"
