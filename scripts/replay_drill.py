#!/usr/bin/env python
"""Rehearse a BTS replay locally, by faking one and submitting it.

Builds an exposure directory from a real exposure: **same pixels, deliberately
wrong metadata**, with `EMUIMAGE` stamped so the server can find the original.
Then it prints the `/submit` call to make against it.

The mangling is the point. Copying an exposure unmodified would pass whether or
not substitution works, because the live headers and the substituted ones would
be identical -- so the drill would prove nothing. Here the filter, pointing,
time and focus are all moved somewhere obviously wrong, and the service is
correct only if it reports the *original* values back.

That matters more than it sounds: an unsubstituted replay does not fail. It
comes back DONE with a plausible-looking fit, because every cross-check
downstream tests consistency rather than truth (see `replay`). So this script
prints the expected answer before anything runs, and you compare against that
rather than deciding afterwards whether the output looked about right.

    scripts/replay_drill.py                               # first in the tree
    scripts/replay_drill.py --source replay/MC_O_20260713_000250

Contacts no server itself, so it needs no host and no token: it leaves a
directory and prints the request. What it is for is the part no test can reach
-- that substituted headers survive assembly and come back as the original
exposure's Zernikes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from lsst.ts.donut_server import client
from lsst.ts.donut_server import daq
from lsst.ts.donut_server import endpoint
from lsst.ts.donut_server import replay

# Pulled off a DAQ filename to name the file it is written back out as. What
# detector a file *is* gets read from its header, in `daq`.
_SENSOR_RE = re.compile(r"_(R\d\d_SW\d)\.fits$")

# Where the fake replay's metadata is moved to. Far from any real pointing, in
# a filter the service *does* have calibs for, and at a different epoch. The
# filter has to be a known one: an unknown filter is rejected at resolution, so
# the drill would fail for the wrong reason and prove nothing about it.
FAKE_OBSID = "BT_S_20261023_000007"
FAKE_DAYOBS = "20261023"
FAKE_SEQNUM = 7
FAKE_RA = 120.0
FAKE_DEC = -45.0
FAKE_DATE_BEG = "2026-10-23T01:00:00.000"
FAKE_DATE_END = "2026-10-23T01:00:30.000"
# The two dates above, as MJD. Hardcoded rather than derived, to keep
# astropy.time off this path: `Time(FAKE_DATE_BEG, scale="tai").mjd` is where
# they came from, and is what to re-run if either date changes. TAI because
# that is what these headers' TIMESYS says.
FAKE_MJD_BEG = 61336.041666666664
FAKE_MJD_END = 61336.04201388889


def sensor_of(path: str) -> str:
    """The `R00_SW0`-style sensor in a DAQ filename."""
    m = _SENSOR_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def pick_fake_filter(original: str) -> str:
    """A known physical_filter that is *not* the original's.

    Taken from `client.KNOWN_PHYSICAL_FILTERS` so the mangled value is one the
    service has flats for -- otherwise `_check_known_filter` rejects the
    exposure before substitution is ever reached, and the drill tests nothing.
    """
    candidates = sorted(client.KNOWN_PHYSICAL_FILTERS - {original})
    if not candidates:
        raise RuntimeError(
            f"no known filter other than {original!r} to mangle to; "
            f"known: {sorted(client.KNOWN_PHYSICAL_FILTERS)}"
        )
    return candidates[0]


def first_exposure(replay_dir: str) -> str:
    """The first exposure subdirectory in the replay tree, sorted."""
    if not os.path.isdir(replay_dir):
        raise RuntimeError(f"not a directory: {replay_dir!r}")
    names = sorted(
        entry.name for entry in os.scandir(replay_dir) if entry.is_dir()
    )
    if not names:
        raise RuntimeError(f"{replay_dir!r} has no exposure subdirectories")
    return os.path.join(replay_dir, names[0])


def build_fake_exposure(source: str, target: str) -> dict:
    """Write a mangled copy of `source` into `target`. Returns a report dict.

    One `.fits` plus one `.json` sidecar per corner sensor, the sidecar being
    required beside every corner-sensor file. The sidecar is written from the
    rewritten header, so the two agree -- the real camera's script derives one
    from the other, and a drill whose sidecar disagreed with its FITS would be
    testing a situation that cannot occur.

    `EMUIMAGE` is stamped with the source exposure's own OBSID, which is what a
    real replay of it would carry and what makes the lookup resolve.
    """
    from astropy.io import fits

    paths = daq.sw_paths(source)
    original_header = daq.read_header(paths[0])
    original_filter = str(original_header["FILTER"])
    fake_filter = pick_fake_filter(original_filter)
    fake_band = fake_filter.split("_")[0]
    emuimage = os.path.basename(os.path.normpath(source))

    os.makedirs(target, exist_ok=True)
    written = []
    for path in paths:
        # Keep the sensor suffix, swap the exposure identity, so the filenames
        # look like what BTS would have written.
        out_name = f"{FAKE_OBSID}_{sensor_of(path)}.fits"
        out = os.path.join(target, out_name)

        with fits.open(path) as hdul:
            header = hdul[0].header
            header["OBSID"] = FAKE_OBSID
            header["DAYOBS"] = FAKE_DAYOBS
            header["SEQNUM"] = FAKE_SEQNUM
            header[replay.EMUIMAGE_CARD] = emuimage
            header["FILTER"] = fake_filter
            header["FILTBAND"] = fake_band
            for card, value in (
                ("RA", FAKE_RA), ("RASTART", FAKE_RA), ("RAEND", FAKE_RA),
                ("DEC", FAKE_DEC), ("DECSTART", FAKE_DEC), ("DECEND", FAKE_DEC),
                ("DATE-BEG", FAKE_DATE_BEG), ("DATE-OBS", FAKE_DATE_BEG),
                ("DATE-END", FAKE_DATE_END),
                ("MJD-BEG", FAKE_MJD_BEG), ("MJD-OBS", FAKE_MJD_BEG),
                ("MJD-END", FAKE_MJD_END),
            ):
                header[card] = value
            header["FILENAME"] = out_name
            hdul.writeto(out, overwrite=True)

            # The sidecar the camera's script writes: the primary header as a
            # flat JSON dict, which is what the real ones are.
            sidecar = {"__CONTENT__": "metadata"}
            for key in header:
                if key in ("COMMENT", "HISTORY", ""):
                    continue
                value = header[key]
                sidecar[key] = value if not hasattr(value, "item") else value.item()
            with open(out[: -len(".fits")] + ".json", "w") as f:
                json.dump(sidecar, f, indent=2)
        written.append(out)

    return {
        "written": written,
        "emuimage": emuimage,
        "original_filter": original_filter,
        "original_ra": float(original_header["RA"]),
        "original_dec": float(original_header["DEC"]),
        "original_imgtype": str(original_header["IMGTYPE"]),
        "fake_filter": fake_filter,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--replay-dir",
        default=os.environ.get(replay.REPLAY_DIR_VAR, "replay"),
        help=f"the replay tree, holding one subdirectory per original exposure "
        f"(default: ${replay.REPLAY_DIR_VAR}, else 'replay')",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="the exposure directory to copy pixels from (default: the first "
        "subdirectory of --replay-dir)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="the directory to build the fake replay in, and then submit "
        f"(default: $TMPDIR/replay_drill/{FAKE_OBSID})",
    )
    args = parser.parse_args()

    try:
        source = args.source or first_exposure(args.replay_dir)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not os.path.isdir(source):
        parser.error(f"--source is not a directory: {source!r}")

    target = args.target or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "replay_drill", FAKE_OBSID
    )
    # Refused rather than rebuilt: writing into a directory that already holds
    # an exposure would leave whichever sensors this one does not overwrite,
    # and a nine-sensor directory is a confusing way to find that out.
    if os.path.isdir(target):
        parser.error(
            f"target already exists: {target!r}. Remove it, or pass --target "
            "somewhere else."
        )

    print(f"  source pixels : {source}")
    print(f"  replay tree   : {args.replay_dir}")
    print(f"  target        : {target}")

    try:
        report = build_fake_exposure(source, target)
    except Exception as exc:
        print(f"\nbuild failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"\n  wrote {len(report['written'])} sensors + sidecars")
    print(f"  {replay.EMUIMAGE_CARD:14s}: {report['emuimage']}")

    print("\n  mangled, so that an unsubstituted run is distinguishable:")
    print(f"    FILTER    {report['original_filter']:24s} -> {report['fake_filter']}")
    print(f"    RA        {report['original_ra']:<24.4f} -> {FAKE_RA}")
    print(f"    DEC       {report['original_dec']:<24.4f} -> {FAKE_DEC}")
    print(f"    OBSID     {report['emuimage']:24s} -> {FAKE_OBSID}")

    print("\n  EXPECT -- these are the ORIGINAL values:")
    print(f"          filter={report['original_filter']}  "
          f"imgtype={report['original_imgtype'].lower()}")
    print(f"          boresight=({report['original_ra']:.4f}, "
          f"{report['original_dec']:.4f})")
    print(f"  EXPECT, with {replay.REPLAY_DIR_VAR} unset: a hard FAILURE --")
    print(f"          {replay.EMUIMAGE_CARD} says these are replayed pixels and")
    print("          there is nowhere to read the original's headers from.")

    print("\n" + "=" * 72)
    print("Submit it, and check the reply against the EXPECT block above")
    print("=" * 72)
    print(f"curl -sS -X POST http://127.0.0.1:{endpoint.DEFAULT_PORT}/submit \\")
    print("  -H 'Content-Type: application/json' \\")
    print(f"  --data '{json.dumps({'directory': target})}'")
    print(f"\n  The server needs {replay.REPLAY_DIR_VAR}="
          f"{os.path.abspath(args.replay_dir)}")
    print("  exported where donutServer.py runs -- substitution is server-side,")
    print("  so setting it in this shell does nothing. Unset it there to see a")
    print("  replay refused for having nowhere to read originals from.")
    print('  (Add "dry_run": true to resolve and report without submitting.)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
