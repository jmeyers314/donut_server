#!/usr/bin/env python
"""Submit one exposure to the donut server, waiting for its files to land.

This is the script the camera's per-file hook calls, with the exposure's
directory as its argument. Pass --token explicitly, or export
DONUT_SERVER_TOKEN to match whatever donutServer.py printed. All arguments are
handled by trigger.main().
"""
import os
import sys

# Before importing anything that reaches numpy: keep BLAS/OpenMP
# single-threaded, because the encode step forks a worker per sensor. For fork
# safety, not speed -- this numpy links OpenBLAS built with USE_OPENMP, and
# forking with live OpenMP threads is undefined behaviour. `daq.py` sets the
# same five above its own numpy-reaching imports, for in-process callers that
# never run this script; `setdefault` makes the overlap harmless, and setting
# them here is what guarantees the ordering for the camera's hook.
# daq._assert_single_threaded_blas() verifies it before the first fork.
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

from lsst.ts.donut_server import trigger  # noqa: E402 -- must follow the env block

if __name__ == "__main__":
    sys.exit(trigger.main())
