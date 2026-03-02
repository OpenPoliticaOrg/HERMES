#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROOT_DIR}/tools/context_markov_smoke.py" \
  --repo-root "${ROOT_DIR}" \
  --taxonomy "${ROOT_DIR}/data/taxonomy/example_event_taxonomy.json" \
  --observation "${ROOT_DIR}/data/taxonomy/example_observation_classifiers.json" \
  --markov "${ROOT_DIR}/data/taxonomy/example_markov_chain.json"
