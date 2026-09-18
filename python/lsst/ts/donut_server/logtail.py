"""Bounded reads of the pipeline log, for the monitoring dashboard.

Imported by both processes: the coordinator for `log_path` (it owns the writer),
the front-end for the read helpers. Stdlib only, and deliberately so -- importing
this at coordinator.py module scope must not pull in numpy, which would land
ahead of the env block that clamps OpenBLAS to one thread.

Every read here is capped. The log is opened in append mode and never rotated
(coordinator.py:608), and one job emits ~40 KB of pipeline commentary at a ~30 s
cadence, so the file a dashboard polls is unbounded in a way its responses must
not be.
"""
from __future__ import annotations

import os
import re

DEFAULT_TAIL_BYTES = 64 * 1024
MIN_TAIL_BYTES = 4 * 1024
MAX_TAIL_BYTES = 1024 * 1024

# How far back a per-job slice looks for its marker. ~200 jobs at 40 KB each.
SLICE_SEARCH_BYTES = 8 * 1024 * 1024
# Cap on one job's returned slice: a crash-looping job emits far more than 40 KB.
SLICE_MAX_BYTES = 512 * 1024

# LSST tasks colourize their own message text and configure_logging does not
# strip it, so the file holds SGR sequences that render as literal "[1m[32m" in
# a <pre>.
_ANSI = re.compile(rb"\x1b\[[0-9;]*m")

# The only per-job delimiter in the file (coordinator.py:472). The task's own
# logging knows nothing about jobs.
_MARKER_SUFFIX = b": starting"
# A restart also ends a job's output (coordinator.py:659).
_RESTART_MARKER = b"coordinator ready, pid"


def log_path() -> str:
    """Where the coordinator writes, and where the dashboard reads.

    `or` rather than a dict default, so an explicitly empty DONUT_SERVER_LOG
    still falls back rather than resolving to the cwd.
    """
    return os.environ.get("DONUT_SERVER_LOG") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "donut_server.log"
    )


def _clean(data: bytes) -> str:
    """Strip SGR escapes, then decode tolerantly.

    errors="replace" is required, not defensive: a byte-range read can split a
    multi-byte sequence at either end of the window.
    """
    return _ANSI.sub(b"", data).decode("utf-8", errors="replace")


def _read_tail(n_bytes: int) -> tuple[str, int, int, bytes]:
    """Read the last `n_bytes` of the log. Returns (path, size, start, data).

    fstat is taken from the open fd rather than the path, so size and content
    come from the same inode even if the file is replaced mid-read.
    """
    path = log_path()
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        start = max(0, size - n_bytes)
        f.seek(start)
        return path, size, start, f.read(n_bytes)


def tail(n_bytes: int = DEFAULT_TAIL_BYTES) -> dict:
    """The last `n_bytes` of the log, clipped to whole lines."""
    n_bytes = max(MIN_TAIL_BYTES, min(MAX_TAIL_BYTES, n_bytes))
    try:
        path, size, start, data = _read_tail(n_bytes)
    except FileNotFoundError:
        # The coordinator only configures logging after its hello, so there is a
        # ~15 s window on a fresh checkout where this is the normal answer.
        return {"path": log_path(), "exists": False}

    if start > 0:
        # Otherwise the first line is a mid-line fragment.
        nl = data.find(b"\n")
        data = data[nl + 1 :] if nl >= 0 else b""

    return {
        "path": path,
        "exists": True,
        "size": size,
        "start": start,
        "returned_bytes": len(data),
        "truncated": start > 0,
        "text": _clean(data),
    }


def job_slice(job_id: str) -> dict:
    """The log lines one job produced, cut between its marker and the next.

    `job_id` reaches neither the filesystem nor a regex -- it is only ever a
    literal needle for bytes.rfind -- so it needs no validation.
    """
    try:
        path, size, start, data = _read_tail(SLICE_SEARCH_BYTES)
    except FileNotFoundError:
        return {"path": log_path(), "exists": False, "job_id": job_id, "found": False}

    base = {
        "path": path,
        "exists": True,
        "size": size,
        "job_id": job_id,
        "searched_bytes": len(data),
    }

    needle = b"job " + job_id.encode() + _MARKER_SUFFIX
    # The last occurrence: job ids are uuid4 (server.py:725), so this is
    # unambiguous even across runs appended to the same file.
    i = data.rfind(needle)
    if i < 0:
        # Not an error: any job older than the search window has no slice, and an
        # operator hits that routinely.
        return {**base, "found": False}

    begin = data.rfind(b"\n", 0, i) + 1
    after = i + len(needle)
    end = data.find(_MARKER_SUFFIX, after)
    if end < 0:
        end = data.find(_RESTART_MARKER, after)
    if end < 0:
        end = len(data)
    else:
        end = data.rfind(b"\n", 0, end) + 1

    truncated = end - begin > SLICE_MAX_BYTES
    if truncated:
        end = begin + SLICE_MAX_BYTES

    return {
        **base,
        "found": True,
        "start": start + begin,
        "end": start + end,
        "returned_bytes": end - begin,
        "truncated": truncated,
        "text": _clean(data[begin:end]),
    }
