#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python tools/soc_dashboard_client.py "$@"
