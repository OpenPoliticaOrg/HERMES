#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python tools/soc_services_smoke.py "$@"
