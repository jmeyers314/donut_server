#!/usr/bin/env python
"""Drive the service over a directory tree of DAQ-native LSSTCam exposures.

**Temporary.** This exists because the DAQ writes *unassembled amplifier*
images -- one FITS per corner sensor with eight `Segment1x` HDUs -- and the
service's wire format carries an assembled `ExposureF`. Assembly belongs on the
producer side eventually (or the raws get ingested into a butler repo and
`donutClient.py -b` handles them); until then it happens in `daq.py`,
in-process, with `lsst.obs.lsst`.

This script is the fixed-cadence, operator-driven producer: it walks a tree
that is already complete and submits every matching exposure on a chosen
period. The camera-triggered producer that waits for one exposure's files to
land is `donutTrigger.py`; both read exposure directories through `daq.py`.

Layout it expects, one directory per exposure, as the replay tree is laid out:

    <root>/MC_O_20260713_000050/MC_O_20260713_000050_R00_SW0.fits
                               /MC_O_20260713_000050_R00_SW0.json
                               /..._R00_SG0_guider.fits          # ignored

Only `_R??_SW?.fits` is read: SG* are guider sensors, not corner wavefront
sensors, and the service has no calibs for them.

Every matched exposure is submitted regardless of `IMGTYPE`, which is
deliberate -- such a tree mixes CWFS, ACQ, science pairs and a true dark, and
watching the service answer an off-nominal frame is part of the point. A
non-CWFS exposure is flagged in its banner so a nonsense result is
attributable at a glance.

    scripts/run_sequence.py --delay 60
    scripts/run_sequence.py --glob 'MC_O_20260713_*?50' --delay 39
    scripts/run_sequence.py --delay 0 --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time

import requests

from lsst.ts.donut_server import client
from lsst.ts.donut_server import daq



def discover_dirs(root: str, pattern: str) -> list[str]:
    """Exposure directories under `root` whose basename matches `pattern`.

    Matched against the basename, not the full path, so a pattern stays short
    (`'*?50'`) and means the same thing whatever `--root` is. Sorted, which for
    `MC_<controller>_<dayobs>_<seqnum>` names is chronological order.
    """
    if not os.path.isdir(root):
        raise RuntimeError(f"--root is not a directory: {root!r}")
    found = [
        entry.path
        for entry in os.scandir(root)
        if entry.is_dir() and fnmatch.fnmatch(entry.name, pattern)
    ]
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        default="replay",
        help="directory holding one subdirectory per exposure (default: replay)",
    )
    parser.add_argument(
        "-g",
        "--glob",
        default="MC_O_*",
        help="shell pattern matched against exposure directory *basenames*, "
        "e.g. 'MC_O_20260713_*?50' (default: MC_O_*). Quote it, or the shell "
        "will expand it first.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        required=True,
        help="seconds between exposure *starts*. Interpreted as a cadence "
        "period, so processing time is subtracted: --delay 39 submits every "
        "39 s regardless of how long each push takes. Use 0 for back-to-back.",
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.environ.get("DONUT_SERVER_TOKEN", ""))
    parser.add_argument(
        "--wait", type=float, default=60.0, help="seconds to long-poll /result"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="cores the coordinator forks for each push",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="also fetch the per-donut stamp table for each exposure, saving "
        "<job_id>.parquet in the current directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and report every matched exposure without contacting the "
        "server or reading any pixels",
    )
    # Same -c/-C semantics as donutClient.py: shared dest, so command-line
    # order is preserved and the last write to a field wins. Applied to every
    # exposure in the sequence.
    parser.add_argument(
        "-c",
        "--config",
        dest="config_overrides",
        action=client.ConfigOverrideAction,
        metavar="FIELD=VALUE",
        help="override one DonutBlitzCornerConfig field for every exposure, "
        "e.g. -c maxFitScatter=2.0. Repeatable, may interleave with -C.",
    )
    parser.add_argument(
        "-C",
        "--config-file",
        dest="config_overrides",
        action=client.ConfigOverrideAction,
        metavar="FILE",
        help="apply a config override file to every exposure (its contents are "
        "sent). Repeatable.",
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be >= 0")
    if args.num_workers < 1:
        parser.error("-n/--num-workers must be >= 1")
    # Checked before any resolution, so a remote run without a token fails now
    # rather than after the first exposure's headers are read.
    if not args.dry_run and not args.token and not client.host_is_loopback(args.host):
        parser.error(
            f"--token is required for a remote --host ({args.host}); "
            "or set DONUT_SERVER_TOKEN"
        )

    directories = discover_dirs(args.root, args.glob)
    if not directories:
        parser.error(
            f"--glob {args.glob!r} matched no directories in {args.root!r}"
        )
    print(f"matched {len(directories)} exposure(s) in {args.root!r}")

    failures: list[str] = []
    for index, directory in enumerate(directories, start=1):
        name = os.path.basename(directory)
        started = time.monotonic()

        print(f"\n=== [{index}/{len(directories)}] {name} ===")
        try:
            source, image_type = daq.resolve_exposure_dir(directory)
        except Exception as exc:
            # Resolution failures are per-exposure too: one directory with a
            # short sensor list should not end an overnight sequence.
            print(f"  resolve failed: {exc}")
            failures.append(name)
            continue

        if image_type.upper() not in ("CWFS", "SCIENCE", "ACQ"):
            print(
                f"  note: IMGTYPE {image_type!r} is not one of CWFS, SCIENCE, "
                "ACQ, so treat its Zernikes with suspicion"
            )

        if args.dry_run:
            print(
                f"  would submit visit={source.visit} "
                f"filter={source.physical_filter} "
                f"detectors={sorted(source.handles)} "
                f"calib_time={source.calib_time}"
            )
        else:
            try:
                client.run_once(
                    args.host,
                    args.token,
                    source,
                    args.wait,
                    images=args.images,
                    config_overrides=args.config_overrides,
                    num_workers=args.num_workers,
                )
            except requests.HTTPError as exc:
                # Deliberately broader than donutClient.py's --loop, which
                # swallows only 409/503: here *any* single-exposure failure is
                # worth surviving, since the sequence is the experiment and a
                # dark or an in-focus frame may well be refused on its merits.
                print(f"  push failed: {exc}")
                failures.append(name)
            except requests.RequestException as exc:
                print(f"  request failed: {exc}")
                failures.append(name)
            except Exception as exc:
                print(f"  failed: {type(exc).__name__}: {exc}")
                failures.append(name)

        if index == len(directories):
            break
        elapsed = time.monotonic() - started
        remaining = args.delay - elapsed
        if remaining > 0:
            print(f"  sleeping {remaining:.1f}s to hold a {args.delay:g}s cadence")
            time.sleep(remaining)
        elif args.delay:
            # Worth saying out loud: the service is not keeping up with the
            # requested cadence, which is usually the thing being measured.
            print(
                f"  behind cadence by {-remaining:.1f}s "
                f"({elapsed:.1f}s elapsed > {args.delay:g}s) -- continuing immediately"
            )

    done = len(directories) - len(failures)
    print(f"\n{done}/{len(directories)} exposure(s) succeeded")
    if failures:
        print(f"failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
