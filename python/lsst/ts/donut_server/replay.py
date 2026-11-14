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
directory holding two different replays is otherwise completely silent.

**The marker is the switch, not the variable.** The keyword is absent entirely
on real camera files, so its presence is a reliable statement that these pixels
are a replay, and that is what substitution is gated on:

- marker present, `DONUT_SERVER_REPLAY_DIR` set -- substitute;
- marker present, variable unset -- hard error. These are replayed pixels and
  there is nowhere to read the original's headers from, so the only
  alternative is to submit them under BTS's pointing, which is the silent
  wrong answer above;
- marker absent -- no substitution, and the variable does not matter.

That last case is what lets the variable be part of a standing test-stand
setup: production is the marker-absent path whether or not some tree is
configured, so the two deployments need not differ in anything but their data.

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
# camera files, which is what makes its presence the switch for substitution.
EMUIMAGE_CARD = "EMUIMAGE"

REPLAY_DIR_VAR = "DONUT_SERVER_REPLAY_DIR"


def read_emuimage(headers: dict) -> str | None:
    """The replayed exposure's OBSID, or `None` if these are not replays.

    Read from all 8 sensors and cross-checked rather than trusted from one. A
    directory holding files from two different replays is otherwise completely
    silent: the substitution would succeed, every downstream cross-check would
    pass, and the answer would be a fit of one exposure's donuts against
    another's calibs.

    Absent on *every* sensor is the real-camera answer -- the keyword is
    written only by the emulator -- and returns `None`, which is what gates
    substitution off. Absent on only some is a hard error: one directory
    cannot be both, so it is either a mix of replayed and real files or a
    replay whose marker was not written everywhere, and neither can be
    resolved by guessing. A blank value counts as absent, being the plausible
    shape of a keyword the emulator declared but never filled.

    Parameters
    ----------
    headers : dict
        {detector name: primary header}, as `read_live_headers` returns it.

    Returns
    -------
    str or None
        The OBSID named by `EMUIMAGE`, or `None` if no sensor carries one.

    Raises
    ------
    RuntimeError
        If only some sensors carry a marker, or if they disagree.
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

    if not values:
        # Real camera data: no marker anywhere, so nothing to substitute.
        return None
    if missing:
        carried = sorted(det for dets in values.values() for det in dets)
        raise RuntimeError(
            f"{EMUIMAGE_CARD} is absent or blank on detector(s) {missing} but "
            f"present on {carried}, so the exposure being replayed cannot be "
            f"identified. One directory cannot be both a replay and real "
            f"camera data: either its files come from two different sources, "
            f"or the emulator did not write the marker on every sensor."
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
            f"not in it."
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
    """Substitute headers for `directory`, or None if it is not a replay.

    `None` means "no substitution" and is the real-camera path: these files
    carry no `EMUIMAGE`, so the headers on disk are the truth and
    `daq.resolve_exposure_dir` wants `headers=None`.

    `report` is called with the lines from `substitution_report`, so the caller
    decides where they land. Omitted means they are not emitted at all, which
    only a caller that has already reported them should do.

    Parameters
    ----------
    directory : str
        The exposure directory, holding one DAQ file per corner sensor.
    report : callable, optional
        Called once per line of human-readable report.

    Returns
    -------
    dict or None
        {detector name: primary header} to substitute, or `None` for real
        camera data.

    Raises
    ------
    RuntimeError
        If these are replays but `DONUT_SERVER_REPLAY_DIR` is unset or does
        not name a directory.

    Notes
    -----
    The live headers are read before the variable is consulted, because the
    marker decides and the variable only says where to look. Production
    therefore pays this read on every exposure to learn that it is not a
    replay. `daq.resolve_exposure_dir` re-reads the same primary headers
    immediately afterwards, so the fix is to share the map rather than to skip
    the read.
    """
    live = read_live_headers(daq.sw_paths(directory))
    exposure = read_emuimage(live)
    if exposure is None:
        return None

    replay_dir = os.environ.get(REPLAY_DIR_VAR)
    if not replay_dir:
        raise RuntimeError(
            f"{EMUIMAGE_CARD} names original exposure {exposure}, so these are "
            f"replayed pixels, but {REPLAY_DIR_VAR} is unset so the original's "
            f"headers cannot be read. Running anyway would fit these donuts "
            f"under whatever pointing, filter and epoch BTS wrote, which no "
            f"downstream check can detect."
        )
    if not os.path.isdir(replay_dir):
        raise RuntimeError(
            f"{REPLAY_DIR_VAR} is {replay_dir!r}, which is not a directory"
        )
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
