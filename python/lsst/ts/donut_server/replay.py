"""Resolving one exposure directory, with BTS replay metadata substituted.

A BTS replay reproduces an original exposure's *pixels* but not its metadata:
the files carry whatever pointing, filter and epoch BTS was configured with,
which is not the sky those donuts were taken under. The service cannot detect
that on its own -- every cross-check downstream tests consistency rather than
truth, so an unsubstituted replay comes back `DONE` with a plausible-looking
fit. So the original headers are read from a second tree and swapped in.

The emulator leaves a breadcrumb naming what it replayed:

    "OBSID"    : "MC_O_20261022_000001",     # what BTS wrote
    "EMUIMAGE" : "MC_O_20250530_000395",     # what it replayed

so the lookup is a path join under `DONUT_SERVER_REPLAY_DIR`. It is read from
all 8 sensors and cross-checked rather than trusted from one, because a
directory holding two different replays is otherwise completely silent. The
keyword is absent entirely on real camera files, so with the variable set an
absent `EMUIMAGE` is a hard error -- that is what it looks like to point the
replay path at real data. `DONUT_SERVER_REPLAY_OBSID` pins one exposure and
skips the lookup, for hand testing and for a replay whose marker is missing or
names something not in the tree.

This module exists because *both* sides need it. It began in `trigger.py`,
where the producer resolved the exposure before pushing pixels; the
coordinator's `/ingest` path reads the files itself and so has to do the same
substitution, on the server, from the same two rules (see `daq`'s module
docstring):

- the map is keyed by **detector name**, and the name is read from the *live*
  file, so a substituted header never decides which detector we are looking at;
- `detector_num` likewise comes from the live header, because it indexes
  cameraGeom for the amp layout the pixels are in.

Needs the stack (it reads FITS headers), which is why it is not in `trigger.py`
-- that module stays importable without `lsst.afw` so the camera's hook pays
only for Python startup.
"""
from __future__ import annotations

import os
from typing import Any

from lsst.ts.donut_server import daq

# The sensor the camera's per-file hook fires on: the only one guaranteed to be
# on disk when we start, and so the one the substitution report is drawn from.
TRIGGER_SENSOR = "R00_SW0"

# The emulator's record of which original exposure it replayed. Absent on real
# camera files, which is what makes it safe to require when replay is enabled.
EMUIMAGE_CARD = "EMUIMAGE"

REPLAY_DIR_VAR = "DONUT_SERVER_REPLAY_DIR"
REPLAY_OBSID_VAR = "DONUT_SERVER_REPLAY_OBSID"


def read_emuimage(headers: dict) -> str:
    """The original exposure's OBSID, from the live headers' `EMUIMAGE` cards.

    Read from all 8 sensors and cross-checked rather than trusted from one. A
    directory holding files from two different replays is otherwise completely
    silent: the substitution would succeed, every downstream cross-check would
    pass, and the answer would be a fit of one exposure's donuts against
    another's calibs.

    Absent is a hard error, not a fall back to the live metadata. With
    `DONUT_SERVER_REPLAY_DIR` set the caller has asserted these are replays, and
    the keyword is absent only on *real camera* files -- so this firing means the
    replay path is pointed at real data, and continuing would submit it with
    whatever pointing happened to be in the headers. A blank value counts as
    absent, being the plausible shape of a keyword the emulator declared but
    never filled.
    """
    missing = []
    values: dict[str, list[str]] = {}
    for name in sorted(headers):
        try:
            raw = headers[name][EMUIMAGE_CARD]
        except Exception:
            raw = None
        value = str(raw).strip() if raw is not None else ""
        if not value:
            missing.append(name)
        else:
            values.setdefault(value, []).append(name)

    if missing:
        raise RuntimeError(
            f"{EMUIMAGE_CARD} is absent or blank on detector(s) "
            f"{missing}, so the exposure being replayed cannot be identified. "
            f"{EMUIMAGE_CARD} is written by the emulator and absent on real "
            f"camera data -- if this is real data, unset {REPLAY_DIR_VAR}; if "
            f"it is a replay, set {REPLAY_OBSID_VAR} to pin the exposure."
        )
    if len(values) != 1:
        detail = "; ".join(
            f"{value!r} on {sorted(dets)}" for value, dets in sorted(values.items())
        )
        raise RuntimeError(
            f"the 8 sensors disagree about {EMUIMAGE_CARD}, so this directory "
            f"holds more than one replayed exposure: {detail}"
        )

    (exposure,) = values
    return exposure


def replay_headers(replay_dir: str, exposure: str) -> dict:
    """{detector name: primary header} for one exposure in the replay tree.

    Keyed by detector name read from each replay header, which is how
    `daq.resolve_exposure_dir` matches a substitute to a live file -- never by
    filename, since the two trees need not name files the same way.
    """
    directory = os.path.join(replay_dir, exposure)
    if not os.path.isdir(directory):
        raise RuntimeError(
            f"the replay names exposure {exposure!r} but {directory!r} is not a "
            f"directory, so its original metadata cannot be read. Either "
            f"{REPLAY_DIR_VAR} points at the wrong tree, or that exposure is "
            f"not in it; {REPLAY_OBSID_VAR} overrides the name."
        )

    headers = {}
    for path in daq.sw_paths(directory):
        header = daq.read_header(path)
        name = daq.detector_name(header)
        if name in headers:
            raise RuntimeError(f"{directory!r} has two files for detector {name}")
        headers[name] = header
    if len(headers) != daq.EXPECTED_SENSORS:
        raise RuntimeError(
            f"replay exposure {directory!r} has {len(headers)} corner sensor(s), "
            f"expected {daq.EXPECTED_SENSORS}: {sorted(headers)}"
        )
    return headers


def read_live_headers(paths: list[str]) -> dict:
    """{detector name: primary header} for the polled directory.

    Keyed by the header's own `RAFTBAY`/`CCDSLOT` rather than by filename, the
    same way `replay_headers` keys the substitutes -- which is what lets the two
    maps be matched detector-wise in `daq.resolve_exposure_dir` without either
    tree's naming mattering.
    """
    headers = {}
    for path in paths:
        header = daq.read_header(path)
        name = daq.detector_name(header)
        if name in headers:
            raise RuntimeError(f"two files claim to be detector {name}")
        headers[name] = header
    return headers


def _card(header, key, default="<absent>"):
    """One header card, for the live-vs-substituted report. Never raises."""
    try:
        value = header[key]
    except Exception:
        return default
    return default if value is None else value


def substitution_report(live, substituted) -> list[str]:
    """Live vs substituted values for the cards that change the answer, as lines.

    These are what a wrong substitution corrupts: the filter picks the flat,
    RA/DEC picks the refcat shards, DATE-BEG picks the calibs, and FOCUSZ is the
    defocus the fit is solving against.

    This report is not a nicety. A replay matched to the wrong exposure still
    produces a DONE job whose summary looks reasonable -- the cross-checks
    downstream test consistency rather than truth -- so these lines are the
    *only* place the substitution is visible for review. Returned rather than
    logged so each caller can put them where its own output goes: the trigger
    prints them on stdout, the coordinator logs them.

    `EMUIMAGE` is reported on its own line above the table, against the
    substituted OBSID, because it does not belong in a live-vs-substituted
    column: the *original* has no EMUIMAGE -- only replays do -- so pairing them
    would show every correct substitution as a change to `<absent>`. What is
    worth checking is that the marker and the exposure actually loaded agree.
    """
    marker = _card(live, EMUIMAGE_CARD)
    loaded = _card(substituted, "OBSID")
    agree = "" if str(marker) == str(loaded) else "   <- MISMATCH"

    lines = [f"replay: {EMUIMAGE_CARD} {marker} -> loaded {loaded}{agree}"]
    lines.append("replay metadata substitution (live -> substituted):")
    for key in ("OBSID", "FILTER", "RA", "DEC", "DATE-BEG", "FOCUSZ"):
        before, after = _card(live, key), _card(substituted, key)
        flag = "" if before == after else "   *"
        lines.append(f"    {key:9s} {before!s:34s} -> {after!s}{flag}")
    return lines


def substituted_headers(directory: str, report=None) -> dict[str, Any] | None:
    """The substitute header map for `directory`, or None if replay is off.

    `None` means "no substitution" and is the real-camera path: with
    `DONUT_SERVER_REPLAY_DIR` unset the headers on disk are the truth, and
    `daq.resolve_exposure_dir` wants `headers=None`.

    `report` is called with the lines from `substitution_report`, so the caller
    decides where they land. Omitted means they are not emitted at all, which
    only a caller that has already reported them should do.
    """
    replay_dir = os.environ.get(REPLAY_DIR_VAR)
    if not replay_dir:
        return None

    if not os.path.isdir(replay_dir):
        raise RuntimeError(
            f"{REPLAY_DIR_VAR} is {replay_dir!r}, which is not a directory"
        )

    live = read_live_headers(daq.sw_paths(directory))

    pinned = os.environ.get(REPLAY_OBSID_VAR)
    if pinned:
        if report is not None:
            report(
                f"replay: {REPLAY_OBSID_VAR}={pinned} pins the exposure, "
                f"ignoring {EMUIMAGE_CARD}"
            )
        exposure = pinned
    else:
        exposure = read_emuimage(live)
        if report is not None:
            report(f"replay: {EMUIMAGE_CARD} names original exposure {exposure}")

    headers = replay_headers(replay_dir, exposure)
    if report is not None:
        # Reported off the trigger sensor, which every exposure has and whose
        # cards are the ones quoted in the module docstring.
        for line in substitution_report(
            live.get(TRIGGER_SENSOR, {}), headers[TRIGGER_SENSOR]
        ):
            report(line)
    return headers


def resolve(directory: str, report=None):
    """`(RawSource, imgtype)` for `directory`, with replay metadata substituted.

    Reads no pixels: everything needed is in the primary headers, which is what
    keeps `/prepare` off the pixel path.
    """
    return daq.resolve_exposure_dir(
        directory, headers=substituted_headers(directory, report=report)
    )
