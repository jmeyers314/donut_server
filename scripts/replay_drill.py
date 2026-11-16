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

    scripts/replay_drill.py                                  # first exposure in the tree
    scripts/replay_drill.py --source replay/MC_O_20260713_000250
    scripts/replay_drill.py --emuimage MC_O_99999999_000001  # deliberate mismatch

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
import shutil
import sys
import time

from lsst.ts.donut_server import client
from lsst.ts.donut_server import daq
from lsst.ts.donut_server import endpoint
from lsst.ts.donut_server import replay

# Pulled off a DAQ filename to name the sensor in this drill's own progress
# lines. What detector a file *is* gets read from its header, in `daq`.
_SENSOR_RE = re.compile(r"_(R\d\d_SW\d)\.fits$")

# Where the fake replay's metadata is moved to. Far from any real pointing, in a
# filter the service *does* have calibs for, and at a different epoch. The filter
# has to be a known one: an unknown filter is rejected at resolution, so the
# drill would fail for the wrong reason and prove nothing about substitution.
FAKE_OBSID = "BT_S_20261023_000007"
FAKE_DAYOBS = "20261023"
FAKE_SEQNUM = 7
FAKE_RA = 120.0
FAKE_DEC = -45.0
FAKE_DATE_BEG = "2026-10-23T01:00:00.000"
FAKE_DATE_END = "2026-10-23T01:00:30.000"
FAKE_MJD_BEG = 61336.041666666664
FAKE_MJD_END = 61336.04201388889
FAKE_FOCUSZ = 0.0


def sensor_of(path: str) -> str:
    """The `R00_SW0`-style sensor in a DAQ filename, for progress reporting."""
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


def build_fake_exposure(source: str, staging: str, emuimage: str) -> dict:
    """Write a mangled copy of `source` into `staging`. Returns a report dict.

    One `.fits` plus one `.json` sidecar per corner sensor, the sidecar being
    required beside every corner-sensor file. The sidecar is written from
    the rewritten header, so the two agree -- the real camera's script derives
    one from the other, and a drill whose sidecar disagreed with its FITS would
    be testing a situation that cannot occur.
    """
    from astropy.io import fits

    paths = daq.sw_paths(source)
    original_header = daq.read_header(paths[0])
    original_filter = str(original_header["FILTER"])
    fake_filter = pick_fake_filter(original_filter)
    fake_band = fake_filter.split("_")[0]

    os.makedirs(staging, exist_ok=True)
    written = []
    for path in paths:
        name = os.path.basename(path)
        # Keep the sensor suffix, swap the exposure identity, so the filenames
        # look like what BTS would have written.
        sensor = sensor_of(path)
        out_name = f"{FAKE_OBSID}_{sensor}.fits"
        out = os.path.join(staging, out_name)

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
                ("FOCUSZ", FAKE_FOCUSZ),
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
        "original_obsid": str(original_header["OBSID"]),
        "original_filter": original_filter,
        "original_ra": float(original_header["RA"]),
        "original_dec": float(original_header["DEC"]),
        "original_focusz": float(original_header["FOCUSZ"]),
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
        "--emuimage",
        default=None,
        help=f"the OBSID to stamp into {replay.EMUIMAGE_CARD} (default: the "
        "source exposure's own, which is what makes the lookup succeed). Set it "
        "to something absent to exercise the mismatch error.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="the directory to submit, copied into once the fake is built "
        f"(default: $TMPDIR/replay_drill/{FAKE_OBSID})",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the staging and target directories behind for inspection",
    )
    args = parser.parse_args()

    try:
        source = args.source or first_exposure(args.replay_dir)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not os.path.isdir(source):
        parser.error(f"--source is not a directory: {source!r}")

    scratch = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "replay_drill"
    )
    staging = os.path.join(scratch, f"{FAKE_OBSID}_staged")
    target = args.target or os.path.join(scratch, FAKE_OBSID)

    # The source exposure's own OBSID is the default EMUIMAGE, which is what a
    # real replay of it would carry -- and what makes the lookup resolve.
    emuimage = args.emuimage or os.path.basename(os.path.normpath(source))

    print("=" * 72)
    print("STEP 1: build a fake replayed exposure, and say what to expect")
    print("=" * 72)
    print(f"  source pixels : {source}")
    print(f"  replay tree   : {args.replay_dir}")
    print(f"  {replay.EMUIMAGE_CARD:14s}: {emuimage}")
    print(f"  staging       : {staging}")
    print(f"  target        : {target}")

    for directory in (staging, target):
        if os.path.isdir(directory):
            shutil.rmtree(directory)
    os.makedirs(target, exist_ok=True)

    t0 = time.monotonic()
    try:
        report = build_fake_exposure(source, staging, emuimage)
    except Exception as exc:
        print(f"\nbuild failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"\n  wrote {len(report['written'])} sensors + sidecars "
        f"({time.monotonic() - t0:.1f}s)"
    )

    print("\n  mangled, so that an unsubstituted run is distinguishable:")
    print(f"    FILTER    {report['original_filter']:24s} -> {report['fake_filter']}")
    print(f"    RA        {report['original_ra']:<24.4f} -> {FAKE_RA}")
    print(f"    DEC       {report['original_dec']:<24.4f} -> {FAKE_DEC}")
    print(f"    FOCUSZ    {report['original_focusz']:<24.4f} -> {FAKE_FOCUSZ}")
    print(f"    OBSID     {report['original_obsid']:24s} -> {FAKE_OBSID}")

    expected_mismatch = args.emuimage is not None and not os.path.isdir(
        os.path.join(args.replay_dir, emuimage)
    )
    print()
    if expected_mismatch:
        print("  EXPECT: the submit FAILS, naming the absent replay directory")
        print(f"          {os.path.join(args.replay_dir, emuimage)!r}")
    else:
        print("  EXPECT, with the replay dir set -- these are the ORIGINAL values:")
        print(f"          filter={report['original_filter']}  "
              f"imgtype={report['original_imgtype'].lower()}")
        print(f"          boresight=({report['original_ra']:.4f}, "
              f"{report['original_dec']:.4f})")
        print(f"  EXPECT, with it unset: a hard FAILURE -- {replay.EMUIMAGE_CARD}")
        print("          says these are replayed pixels and there is nowhere to "
              "read")
        print("          the original's headers from.")

    print("\n" + "=" * 72)
    print(f"STEP 2: copying {len(report['written'])} sensors")
    print("=" * 72)
    for index, path in enumerate(sorted(report["written"]), start=1):
        for suffix in (".fits", ".json"):
            src = path[: -len(".fits")] + suffix
            shutil.copy2(src, os.path.join(target, os.path.basename(src)))
        print(f"  [{index}/{len(report['written'])}] {sensor_of(path)} landed")

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

    if not args.keep:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"\n(removed staging {staging}; --keep to retain. The target "
              f"{target} is left in place to submit.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
