#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER_TEX="${1:-${ROOT_DIR}/paper/networked_contextual_hermes.tex}"
OUT_DIR="${2:-${ROOT_DIR}/paper}"

mkdir -p "${OUT_DIR}"

latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="${OUT_DIR}" "${PAPER_TEX}"
echo "Built paper PDF: ${OUT_DIR}/networked_contextual_hermes.pdf"

