#!/usr/bin/env python
"""Drive the service over a directory tree of DAQ-native LSSTCam exposures.

The fixed-cadence, operator-driven counterpart to the camera's hook: it walks a
tree that is already complete and submits every matching exposure on a chosen
period. Like the hook it POSTs the *directory* to `/submit` and the server
reads the pixels itself, in-process, through `daq.py` -- so nothing here
assembles anything or touches `lsst.obs.lsst`.

Two deliberate departures from the hook's request, which is otherwise the same
`{"directory": ...}`:

- `resubmit` is always sent. The hook's deduplication guard exists to stop one
  exposure being submitted twice by a hook that fired twice; this walks a
  sorted list of distinct directories exactly once, and re-running the whole
  sequence is the normal workflow rather than an accident.
- `config_overrides`, `num_workers` and `any_imgtype` are passed through when
  given. The hook has no way to send any of them, by design.

Layout it expects, one directory per exposure, as the replay tree is laid out:

    <root>/MC_O_20260713_000050/MC_O_20260713_000050_R00_SW0.fits
                               /MC_O_20260713_000050_R00_SW0.json
                               /..._R00_SG0_guider.fits          # ignored

Only `_R??_SW?.fits` is read: SG* are guider sensors, not corner wavefront
sensors, and the service has no calibs for them.

Calibration frames come back `SKIPPED` rather than fitted, exactly as they do
in production, and are counted as skipped rather than failed -- such a tree
mixes CWFS, ACQ, science pairs and a true dark. `--any-imgtype` forces them
through, though a frame taken with tracking disabled has no pointing to prepare
against, so forcing one currently fails at `/prepare`. Anything outside
`CWFS`/`ACQ`/`SCIENCE`/`OBJECT` is fitted and flagged in its banner, so a
nonsense result is attributable at a glance.

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
from lsst.ts.donut_server import endpoint

# `/submit` returns only once the fit is DONE, so the result is already written
# by the time it is asked for and this is a guard rather than a long poll.
RESULT_WAIT_S = 10.0

# Covers the server's resolve, prepare and the whole fit, the same span the
# camera hook's `--max-time` covers. A backstop against a wedged server, not a
# cadence control -- `--delay` is that.
SUBMIT_TIMEOUT_S = 900


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


def submit_exposure(
    host: str,
    token: str,
    directory: str,
    config_overrides: list | None = None,
    num_workers: int | None = None,
    any_imgtype: bool = False,
    dry_run: bool = False,
    images: bool = False,
) -> str:
    """POST one exposure directory to `/submit` and report what came back.

    Returns the state the server answered with -- `DONE`, `SKIPPED` or
    `DRY_RUN` -- so the caller can tell a declined calibration frame from a
    failure. Anything else raises, since every other outcome is an HTTP error.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Absolute, because `/submit` requires it: a relative path would resolve
    # against the server's cwd rather than ours and mean something else.
    payload: dict = {"directory": os.path.abspath(directory), "resubmit": True}
    if config_overrides:
        payload["config_overrides"] = config_overrides
    if any_imgtype:
        payload["any_imgtype"] = True
    if dry_run:
        payload["dry_run"] = True

    t0 = time.monotonic()
    resp = requests.post(
        f"{host}/submit",
        json=payload,
        params={"num_workers": num_workers} if num_workers is not None else None,
        headers=headers,
        timeout=SUBMIT_TIMEOUT_S,
    )
    client.raise_for_status(resp)
    body = resp.json()
    state = str(body.get("state", "?"))

    # What the server's own header read found, which is the only description of
    # the exposure there is now that nothing is resolved on this side.
    exposure = body.get("exposure") or {}
    if exposure:
        print(
            f"  exposure -> visit={exposure.get('visit')} "
            f"filter={exposure.get('physical_filter')} "
            f"imgtype={exposure.get('imgtype')} "
            f"detectors={exposure.get('detectors')}"
        )
    if note := body.get("note"):
        print(f"  note: {note}")

    if state == "DRY_RUN":
        return state
    if state == "SKIPPED":
        print(f"  skipped: {body.get('reason')}")
        return state

    job_id = body["job_id"]
    print(
        f"  submit -> state={state} job_id={job_id} "
        f"({time.monotonic() - t0:.3f}s)"
    )
    for label, timings in (
        ("prepare", body.get("prepare_timings")),
        ("submit", body.get("timings")),
    ):
        for key, value in (timings or {}).items():
            print(f"    {label}.{key}: {value}")

    client.report_result(host, headers, job_id, RESULT_WAIT_S, images)
    return state


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
        "39 s regardless of how long each fit takes. Use 0 for back-to-back.",
    )
    parser.add_argument("--host", default=endpoint.DEFAULT_HOST)
    parser.add_argument("--token", default=os.environ.get("DONUT_SERVER_TOKEN", ""))
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=None,
        help="cores the coordinator forks for each exposure (default: the "
        "server's own DONUT_SERVER_NUM_WORKERS)",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="also fetch the per-donut stamp table for each exposure, saving "
        "<job_id>.parquet in the current directory",
    )
    parser.add_argument(
        "--any-imgtype",
        action="store_true",
        help="fit calibration frames too, instead of letting the server "
        "decline them. A frame with tracking disabled has no pointing, so this "
        "currently fails at /prepare for exactly the frames it exists for.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="have the server resolve and report every matched exposure "
        "without preparing or fitting any of them",
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
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("-n/--num-workers must be >= 1")
    # Checked before the first request, so a remote run without a token fails
    # now rather than after the first exposure.
    if not args.token and not client.host_is_loopback(args.host):
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
    skipped: list[str] = []
    for index, directory in enumerate(directories, start=1):
        name = os.path.basename(directory)
        started = time.monotonic()

        print(f"\n=== [{index}/{len(directories)}] {name} ===")
        try:
            state = submit_exposure(
                args.host,
                args.token,
                directory,
                config_overrides=args.config_overrides,
                num_workers=args.num_workers,
                any_imgtype=args.any_imgtype,
                dry_run=args.dry_run,
                images=args.images,
            )
        except requests.HTTPError as exc:
            # Deliberately broader than donutClient.py's --loop, which swallows
            # only 409/503: here *any* single-exposure failure is worth
            # surviving, since the sequence is the experiment and one bad
            # directory should not end an overnight run.
            print(f"  submit failed: {exc}")
            failures.append(name)
        except requests.RequestException as exc:
            print(f"  request failed: {exc}")
            failures.append(name)
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}")
            failures.append(name)
        else:
            if state == "SKIPPED":
                skipped.append(name)

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

    done = len(directories) - len(failures) - len(skipped)
    print(f"\n{done}/{len(directories)} exposure(s) succeeded")
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
    if failures:
        print(f"failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
