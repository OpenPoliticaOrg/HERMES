#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python tools/soc_grpc_smoke.py "$@"
