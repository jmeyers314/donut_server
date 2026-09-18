#!/usr/bin/env bash
# Start the donut_server FastAPI app. Handles .env sourcing (incl. the
# LD_LIBRARY_PATH -> DYLD_LIBRARY_PATH remap SIP strips on macOS) and
# DONUT_SERVER_TOKEN. Any extra args are passed through to uvicorn.
set -euo pipefail
cd "$(dirname "$0")"

set -a
source .env
set +a
export DYLD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Keep every threaded runtime single-threaded across the whole process tree.
# coordinator.py sets these too, but env vars are only read when the libraries
# load, so that only binds if coordinator is imported before numpy; exporting
# here does not depend on import order.
#
# This is for fork safety, not speed: A/B'd at 1 vs 12 threads, per-job wall
# time is a dead heat (6.9 s vs 6.8 s). But numpy here links OpenBLAS built
# with USE_OPENMP, and the task forks 8 workers -- unclamped, each fork happens
# with a live OpenMP thread pool, which is undefined behaviour rather than just
# slower.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Owned here so the path can be announced below; coordinator.py falls back to
# the same default when imported directly (e.g. its own smoke test).
export DONUT_SERVER_LOG="${DONUT_SERVER_LOG:-$PWD/donut_server.log}"
echo "Pipeline logs (INFO and above) -> $DONUT_SERVER_LOG"
echo "  tail -f $DONUT_SERVER_LOG"

if [ -z "${DONUT_SERVER_TOKEN:-}" ]; then
    export DONUT_SERVER_TOKEN="$(openssl rand -hex 16)"
    echo "DONUT_SERVER_TOKEN not set; generated one for this run:"
    echo "  $DONUT_SERVER_TOKEN"
    echo "Use it from the client, e.g.:"
    echo "  ./run_client.sh --token $DONUT_SERVER_TOKEN"
fi

exec uvicorn server:app --host 127.0.0.1 --port 8000 "$@"
