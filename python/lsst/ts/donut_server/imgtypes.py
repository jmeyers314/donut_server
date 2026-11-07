"""Which IMGTYPEs are worth a wavefront fit.

Compared in the front-end, by `/submit`, against the image type a `resolve`
returns. The two sides cannot share a module directly: reading the image type
takes the stack (`daq.resolve_exposure_dir` via the coordinator), and
`server.py` imports no part of it. So the sets live here, where both can reach
them, and this module imports nothing at all.
"""
from __future__ import annotations

# Image types that are never worth a wavefront fit. Calibration frames have no
# stars, so they would burn a full flight to arrive at an empty donut catalog.
SKIP_IMGTYPES = frozenset({"DARK", "BIAS", "FLAT"})

# Types a wavefront fit is meaningful for. Anything else is processed but
# flagged, the same way run_sequence.py flags it.
EXPECTED_IMGTYPES = frozenset({"CWFS", "ACQ", "SCIENCE", "OBJECT"})
