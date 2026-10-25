"""The camera-triggered producer: wait for one exposure's files, then submit it.

In production the camera drives this service. Whenever the camera writes a FITS
file it invokes a per-file script that compresses it, writes a JSON sidecar, and
ships it to OODS and USDF; one extra step in that script calls *this* producer.
The hook is wired for **one sensor only** -- `R00_SW0` -- so that a script that
runs per *file* fires exactly once per *exposure*. It is handed the exposure's
directory.

Firing on the first corner sensor means the other seven may not have landed yet,
so the first thing this does is poll. Only once all 8 `_R??_SW?.fits` are present
and have stopped growing does the /prepare -> /push -> /result cycle run. That
means nothing is prepared while the shutter is open -- everything happens after
readout -- which costs latency the two-phase protocol was built to avoid, but
the service still operates, and the camera hook is the only trigger available
today.

    bin/donutTrigger.py /path/to/MC_O_20260713_000050
    bin/donutTrigger.py /path/to/exposure --detach --log-dir /var/log/aos
    bin/donutTrigger.py /path/to/exposure --dry-run      # resolve, submit nothing

## Replayed exposures (BTS)

The Base Test Stand can replay real images through the simulated camera, and
files appear in a directory exactly as they would on the real camera. But **BTS
replays the pixels, not the metadata**: the headers carry the simulated
telescope's time and pointing, not the values that were present when the data
was taken.

**The service cannot detect this and will not refuse it**, which is why this is
handled here rather than left to fail loudly downstream. Measured: a replayed
exposure pushed under BTS's own headers (`z_20`, boresight (120, -45), a date
three months off, against `r_57` pixels from (215.5, -10.1)) came back DONE
with 74 rows and 6/6 groups "succeeded". Every coordinator cross-check passed,
because each checks *consistency*, not truth -- the filter check compares the
raws against what /prepare loaded and both said `z_20`, and the refcat-coverage
check compares the shards against a WCS built from the same bad boresight, so
they matched exactly. Donuts from one field were fit against another field's
Gaia sources and the fit converged. Nothing anywhere flags it.

Setting `DONUT_SERVER_REPLAY_DIR` switches on substitution: the original
exposure is located in that tree and *its* primary headers are used in place of
what BTS wrote. Unset, headers come from the files in the polled directory,
which is what the real camera needs.

The variable rather than the mere presence of a replay marker is the switch,
deliberately: running a replay *without* substitution has probative value --
forcing every WCS to describe the wrong sky is a real test -- so that has to
stay expressible.

How the original is located -- the `EMUIMAGE` marker, the 8-sensor
cross-check, and `DONUT_SERVER_REPLAY_OBSID` as the override -- lives in
`replay`, because the coordinator's `/ingest` path has to do the same
substitution on the server side. See that module's docstring.

Everything above `resolve_source` here is deliberately free of any `lsst`
import, so the polling logic is testable without the stack -- and so
`--detach`'s parent process can exit before paying for it. `replay` is
imported from inside `resolve_source` for that reason: it reads FITS headers,
and so reaches afw.
"""
from __future__ import annotations

import argparse
import errno
import glob
import os
import re
import subprocess
import sys
import time

# `client` is imported lazily, by each function that needs it, and never at
# module scope. It reaches `lsst.afw.image` through `exposure_codec`, and the
# whole value of `--detach` is that the parent hands control back to the
# camera's hook without paying for that. Measured on a `--detach` parent: 6.1 s
# wall before deferring, 1.7 s after, against a 0.73 s bare-interpreter
# baseline on this filesystem -- i.e. what is left is Python startup, not us.
# `requests` is deferred for the same reason; it is only needed to classify a
# submission that failed. `tests/test_trigger.py` asserts the module stays
# clean of `lsst.afw`, since a stray top-level import would silently undo this.

EXPECTED_SENSORS = 8

# Corner-sensor FITS, and the sidecar the camera's script writes beside it.
SW_GLOB = "*_R??_SW?.fits"
SIDECAR_SUFFIX = ".json"

# Pulled off a DAQ filename to report which sensors are still missing while
# polling. Only ever used for progress reporting and error messages -- what
# detector a file *is* gets read from its header, in daq.py.
_SENSOR_RE = re.compile(r"_(R\d\d_SW\d)\.fits$")

# The replay cards, variables and substitution logic live in `replay`, which
# both this module and the coordinator's /ingest path use. Not re-exported here:
# importing it reaches the stack, and the whole point of this module's lazy
# imports is that `--detach`'s parent never does.
#
# `REPLAY_DIR_VAR` is the one name tests reach for by string, so it is spelled
# out rather than imported: `replay.REPLAY_DIR_VAR` is the definition.
REPLAY_DIR_VAR = "DONUT_SERVER_REPLAY_DIR"

# Set on the detached child so it does not detach again. Not a flag, because the
# child's argv is the parent's argv and rewriting it is more fragile than one
# variable the parent cannot have inherited.
DETACHED_VAR = "DONUT_SERVER_TRIGGER_DETACHED"

# Image types that are never worth a wavefront fit, skipped unless
# --any-imgtype. Calibration frames have no stars, so they would burn a full
# push to arrive at an empty donut catalog.
SKIP_IMGTYPES = frozenset({"DARK", "BIAS", "FLAT"})

# Types a wavefront fit is meaningful for. Anything else is processed but
# flagged, the same way run_sequence.py flags it.
EXPECTED_IMGTYPES = frozenset({"CWFS", "ACQ", "SCIENCE", "OBJECT"})


class ConfigOverrideAction(argparse.Action):
    """`client.ConfigOverrideAction`, imported only if -c/-C is actually used.

    Subclassing it directly would mean importing `client` to *build* the parser,
    which is the one import `--detach`'s parent must not pay for (see the note
    at the imports). Delegating from `__call__` defers it to the run that
    actually passes an override, and keeps one implementation of the -c/-C
    semantics rather than a second copy that can drift from `pipetask`'s.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        from lsst.ts.donut_server import client

        client.ConfigOverrideAction.__call__(
            self, parser, namespace, values, option_string
        )


def log(message: str) -> None:
    """One line, timestamped and flushed.

    Flushed because the detached child's stdout is a file, not a tty, so it is
    block-buffered -- and the whole point of that log is being readable *while*
    a slow poll is still going.
    """
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)


def sensor_of(path: str) -> str:
    """The `R00_SW0`-style sensor in a DAQ filename, for progress reporting."""
    m = _SENSOR_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def poll_once(directory: str) -> dict[str, int]:
    """{sensor: size} for every corner-sensor FITS that has its sidecar.

    A FITS with no sidecar is reported as absent rather than as an error: during
    the window the camera's script is working on a file, that is exactly the
    state on disk, and it is about to fix itself.
    """
    found: dict[str, int] = {}
    for path in glob.glob(os.path.join(directory, SW_GLOB)):
        sidecar = path[: -len(".fits")] + SIDECAR_SUFFIX
        if not os.path.exists(sidecar):
            continue
        try:
            found[sensor_of(path)] = os.path.getsize(path)
        except OSError:
            # Vanished between glob and stat -- a rewrite in progress. Absent
            # for this poll; the next one will see it.
            continue
    return found


def wait_for_exposure(
    directory: str,
    timeout: float = 120.0,
    interval: float = 1.0,
    stable_checks: int = 2,
) -> list[str]:
    """Block until all 8 corner sensors are present and have stopped growing.

    Completeness alone is not enough to start reading: the files are ~9.5 MB
    each and a FITS that exists is not necessarily a FITS that is finished, so
    every size must repeat unchanged across `stable_checks` consecutive polls.
    That is the cheap, portable version of this check -- there is no way to ask
    whether another process still holds a descriptor -- and at a 1 s interval it
    costs one extra second against a 30 s exposure cadence.

    The sidecar's existence is a second, weaker signal, used in `poll_once`: we
    do not control the order the camera's script writes the two, so it cannot be
    relied on as a done-marker on its own.

    Returns the sorted FITS paths. On timeout raises, naming the sensors seen
    and the count still missing -- the message to read when a replay delivers
    only six sensors, which is otherwise a very quiet failure.
    """
    if stable_checks < 1:
        raise ValueError("stable_checks must be >= 1")

    deadline = time.monotonic() + timeout
    previous: dict[str, int] | None = None
    stable = 0
    reported = -1

    while True:
        found = poll_once(directory)

        if len(found) != reported:
            log(
                f"polling {directory}: {len(found)}/{EXPECTED_SENSORS} sensors "
                f"({' '.join(sorted(found)) or 'none yet'})"
            )
            reported = len(found)

        if len(found) == EXPECTED_SENSORS and found == previous:
            stable += 1
            if stable >= stable_checks:
                log(f"complete: {EXPECTED_SENSORS} sensors, sizes stable")
                return sorted(glob.glob(os.path.join(directory, SW_GLOB)))
        else:
            # Any change at all restarts the count, including a file *shrinking*
            # or a ninth appearing: the invariant wanted is "nothing moved for
            # N polls", not "N polls since we first saw 8".
            stable = 0
        previous = found

        if time.monotonic() >= deadline:
            missing = EXPECTED_SENSORS - len(found)
            raise TimeoutError(
                f"waited {timeout:g}s for {directory!r} and still have "
                f"{len(found)}/{EXPECTED_SENSORS} sensors with sidecars "
                f"({missing} missing); saw {sorted(found)}"
            )
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def resolve_source(directory: str, paths: list[str]):
    """`(RawSource, imgtype)` for a complete exposure directory.

    Substitutes the original metadata when `DONUT_SERVER_REPLAY_DIR` is set.
    Reads no pixels on either path: everything needed is in the primary headers,
    which is what keeps /prepare off the pixel path.

    The substitution itself lives in `replay`, because the coordinator's
    `/ingest` path has to do the same thing on the server side. `paths` is now
    only the polled file list this process already had; `replay` re-globs the
    directory for its own use, since its other caller has no such list.
    """
    from lsst.ts.donut_server import replay

    return replay.resolve(directory, report=log)


def claim(state_dir: str, key: str) -> bool:
    """Claim `key` exactly once, across processes. True if we got it.

    The camera's hook can fire twice for one exposure -- a retry, a re-ingest,
    an operator running this by hand over a directory that already went through
    -- and a second submission is not harmless: it burns a flight's worth of
    compute and puts a duplicate exposure on the dashboard for the same pixels.

    `O_CREAT | O_EXCL` because the two firings can be concurrent, which rules
    out checking-then-creating.
    """
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, f"{key}.done")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise
    with os.fdopen(fd, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()}\n")
    return True


def detach(log_dir: str, name: str) -> str:
    """Re-run this same command in a new session, and return the log path.

    For the camera's hook: the per-file script is on the camera's own critical
    path, and this producer blocks for the poll plus a full fit. The parent
    reaches here having imported nothing from the stack, so it returns in tens
    of milliseconds.

    What this costs, and why it is opt-in: the child's exit status reaches
    nobody, so a failure is only visible in the log file. `arm_watchdog` exists
    for that reason -- a detached child that wedges is an orphan nobody is
    watching, so it is given a hard upper bound on its whole life.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}.log")
    env = {**os.environ, DETACHED_VAR: "1"}
    with open(path, "a", buffering=1) as handle:
        subprocess.Popen(
            [sys.executable, *sys.argv],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            cwd=os.getcwd(),
        )
    return path


def arm_watchdog(seconds: float) -> None:
    """Guarantee this process exits within `seconds`, whatever it is doing.

    Only used when detached, where nothing supervises us: no parent is waiting,
    the exit status reaches nobody, and a producer blocked forever on a server
    that never answers would sit in the process table indefinitely. `SIGALRM`
    rather than a watchdog thread because it interrupts a blocking socket read,
    which is the realistic way to wedge here.

    Deliberately generous against the sum of the parts (`--timeout` for the
    poll, then `--wait` for the fit): this is the backstop for "something is
    broken in a way we did not anticipate", not a second cadence control.
    """
    import signal

    def expire(signum, frame):
        raise TimeoutError(
            f"watchdog: still running {seconds:g}s after start, giving up"
        )

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(int(seconds))


def run_trigger(
    directory: str,
    host: str,
    token: str,
    wait: float,
    timeout: float = 120.0,
    interval: float = 1.0,
    stable_checks: int = 2,
    state_dir: str | None = None,
    any_imgtype: bool = False,
    dry_run: bool = False,
    images: bool = False,
    config_overrides: list | None = None,
    num_workers: int | None = None,
) -> int:
    """Poll, resolve, submit. Returns a process exit status.

    0 for a submitted exposure *and* for a deliberate skip (an already-claimed
    directory, a dark): the camera's hook should not see a failure because we
    correctly declined to process a bias frame.
    """
    from lsst.ts.donut_server import client

    started = time.monotonic()
    paths = wait_for_exposure(directory, timeout, interval, stable_checks)

    source, imgtype = resolve_source(directory, paths)
    log(
        f"resolved visit={source.visit} filter={source.physical_filter} "
        f"imgtype={imgtype} detectors={sorted(source.handles)}"
    )

    normalized = imgtype.upper()
    if not any_imgtype and normalized in SKIP_IMGTYPES:
        log(f"skipping: IMGTYPE {imgtype!r} has no stars to fit (--any-imgtype to force)")
        return 0
    if normalized not in EXPECTED_IMGTYPES:
        log(
            f"note: IMGTYPE {imgtype!r} is not one of "
            f"{sorted(EXPECTED_IMGTYPES)}, so treat its Zernikes with suspicion"
        )

    # Claimed after resolution, so a directory that fails to resolve can be
    # retried once whatever was wrong with it is fixed -- but before the push,
    # so a concurrent second firing cannot also submit. The visit is the key
    # rather than the directory name: under replay the directory name is BTS's
    # and the same pixels replayed twice should still be caught.
    if state_dir is not None and not dry_run:
        if not claim(state_dir, str(source.visit)):
            log(
                f"already submitted: {state_dir}/{source.visit}.done exists, so "
                "this exposure has been through once already; not resubmitting"
            )
            return 0

    if dry_run:
        boresight_ra, boresight_dec, calib_time = client.read_prepare_args(source)
        log(
            f"dry run, would submit: visit={source.visit} "
            f"filter={source.physical_filter} "
            f"boresight=({boresight_ra:.4f}, {boresight_dec:.4f}) "
            f"calib_time={calib_time}"
        )
        return 0

    log(f"submitting to {host} after {time.monotonic() - started:.1f}s")
    client.run_once(
        host,
        token,
        source,
        wait,
        images=images,
        config_overrides=config_overrides,
        num_workers=num_workers,
    )
    log(f"done in {time.monotonic() - started:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directory",
        help="the exposure's directory, holding one _R??_SW?.fits plus sidecar "
        "per corner sensor. Polled until all 8 arrive.",
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.environ.get("DONUT_SERVER_TOKEN", ""))
    parser.add_argument(
        "--wait", type=float, default=120.0, help="seconds to long-poll /result"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for all 8 sensors before giving up (default: 120)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="seconds between directory polls (default: 1.0)",
    )
    parser.add_argument(
        "--stable-checks",
        type=int,
        default=2,
        help="consecutive polls a file's size must be unchanged before it counts "
        "as finished being written (default: 2)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="cores the coordinator forks for this push",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="also fetch the per-donut stamp table, saving <job_id>.parquet in "
        "the current directory",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="return immediately, doing the work in a new session with output to "
        "--log-dir. For the camera's hook, which should not block on a fit.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="where --detach writes <exposure>.log, and the default for "
        "--state-dir (default: $TMPDIR/donut_trigger)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="where the once-only marker per exposure is kept, so a hook that "
        "fires twice submits once (default: --log-dir). Pass '' to disable.",
    )
    parser.add_argument(
        "--any-imgtype",
        action="store_true",
        help=f"process even {'/'.join(sorted(SKIP_IMGTYPES))} frames, which are "
        "skipped by default as having no stars to fit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="poll and resolve, report what would be submitted, contact no "
        "server and read no pixels. Run this first on a new deployment.",
    )
    # Same -c/-C semantics as donutClient.py: shared dest, so command-line order
    # is preserved and the last write to a field wins.
    parser.add_argument(
        "-c",
        "--config",
        dest="config_overrides",
        action=ConfigOverrideAction,
        metavar="FIELD=VALUE",
        help="override one DonutBlitzCornerConfig field, e.g. "
        "-c maxFitScatter=2.0. Repeatable, may interleave with -C.",
    )
    parser.add_argument(
        "-C",
        "--config-file",
        dest="config_overrides",
        action=ConfigOverrideAction,
        metavar="FILE",
        help="apply a config override file (its contents are sent). Repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be > 0")
    if args.stable_checks < 1:
        parser.error("--stable-checks must be >= 1")
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("-n/--num-workers must be >= 1")
    if not os.path.isdir(args.directory):
        parser.error(f"not a directory: {args.directory!r}")

    log_dir = args.log_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "donut_trigger"
    )
    # '' disables the guard; None means "follow --log-dir".
    state_dir = log_dir if args.state_dir is None else (args.state_dir or None)

    # Before the token check, and before anything else that would need `client`:
    # every check above this line is pure argparse, so the parent reaches here
    # having imported nothing from the stack and returns in ~50 ms.
    if args.detach and not os.environ.get(DETACHED_VAR):
        # Named for the directory, not the visit: the parent must not read a
        # header to pick a log name, since not reading anything is the point.
        path = detach(log_dir, os.path.basename(os.path.normpath(args.directory)))
        print(f"detached, logging to {path}")
        return 0

    if os.environ.get(DETACHED_VAR):
        # Poll, then fit, then a wide margin for everything unmodelled.
        arm_watchdog(args.timeout + args.wait + 300)

    import requests

    from lsst.ts.donut_server import client

    # Checked before any polling, so a misconfigured hook fails at once rather
    # than after waiting out a 120 s timeout to discover it has no credentials.
    # Under --detach this runs in the child, so the complaint lands in the log
    # file rather than on the hook's stderr -- which is true of every other
    # failure there too, and is the trade --detach is.
    if (
        not args.dry_run
        and not args.token
        and not client.host_is_loopback(args.host)
    ):
        parser.error(
            f"--token is required for a remote --host ({args.host}); "
            "or set DONUT_SERVER_TOKEN"
        )

    try:
        return run_trigger(
            args.directory,
            args.host,
            args.token,
            args.wait,
            timeout=args.timeout,
            interval=args.poll_interval,
            stable_checks=args.stable_checks,
            state_dir=state_dir,
            any_imgtype=args.any_imgtype,
            dry_run=args.dry_run,
            images=args.images,
            config_overrides=args.config_overrides,
            num_workers=args.num_workers,
        )
    except TimeoutError as exc:
        log(f"timed out: {exc}")
        return 1
    except requests.HTTPError as exc:
        # Separated from the RequestException below because they want different
        # operator responses: a 4xx/5xx means the service answered and refused,
        # which is a problem with this exposure or this config, while a bare
        # RequestException means it was never reached.
        log(f"push failed: {exc}")
        return 1
    except requests.RequestException as exc:
        log(f"request failed: {exc}")
        return 1
    except Exception as exc:
        # Everything is caught and logged rather than tracebacked, because when
        # detached the traceback goes to a log file nobody is reading and the
        # exit status goes nowhere at all. The message has to carry the cause.
        log(f"failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
