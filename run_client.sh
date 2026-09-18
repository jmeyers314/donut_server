#!/usr/bin/env bash
# Run the donut_server test client. Handles .env sourcing. Pass --token
# explicitly, or export DONUT_SERVER_TOKEN to match whatever run_server.sh
# printed/used. Any args are passed through to client.py.
set -euo pipefail
cd "$(dirname "$0")"

set -a
source .env
set +a
export DYLD_LIBRARY_PATH="$LD_LIBRARY_PATH"

exec python client.py "$@"
