#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python tools/soc_integration_probe.py "$@"
