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

from lsst.ts.donut_server import protocol

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


def _host_is_loopback(host: str) -> bool:
    """`client.host_is_loopback`, reimplemented without reaching the stack.

    Duplicated rather than imported for one reason: importing `client` is the
    ~1.9 s the notifier path exists to avoid, and this is the only thing it
    would want from there. `client`'s version stays the definition -- it is what
    the resolving path and `run_sequence.py` use -- and both are stdlib-only, so
    they cannot drift in behaviour without one of them being wrong.
    """
    import ipaddress
    import socket
    import urllib.parse

    name = urllib.parse.urlsplit(host).hostname
    if not name:
        return False
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(name, None)
    except socket.gaierror:
        return False
    return bool(infos) and all(
        ipaddress.ip_address(info[4][0]).is_loopback for info in infos
    )


def run_notifier(
    directory: str,
    host: str,
    token: str,
    timeout: float = 120.0,
    interval: float = 1.0,
    stable_checks: int = 2,
    config_overrides: list | None = None,
    num_workers: int | None = None,
    any_imgtype: bool = False,
    dry_run: bool = False,
    resubmit: bool = False,
) -> int:
    """Poll, then hand the directory to the server. Returns an exit status.

    The thin half of this module, and what the camera's hook should call: it
    imports no part of the LSST stack, so the whole process costs ~0.4 s of
    Python startup against ~5 s for the resolving path -- 3 s of that being the
    header stack and 1.9 s afw, which are additive. On the critical path
    between "the files are complete" and "the Zernikes exist", that is the
    single largest thing a producer controls.

    Polling stays here rather than moving with everything else. It is
    stdlib-only and therefore free, and the coordinator's command loop is
    serial -- parking it on a filesystem wait would stall every other command
    on that flight for as long as the camera took to finish writing.

    What the server does instead: resolve the headers (including replay
    substitution), prepare against what it found, and read the pixels. One
    request, because the three steps have to land on the same flight and only
    the server can arrange that.

    `--any-imgtype`, `--dry-run` and the resubmit override are flags here but
    decisions there, for the same reason: all three need the image type or the
    visit, and those come out of the header read. `--images` and `--wait` have
    no notifier equivalent, needing the stack to interpret a result;
    `donutClient.py` keeps them.
    """
    import json
    import urllib.error
    import urllib.request

    started = time.monotonic()
    wait_for_exposure(directory, timeout, interval, stable_checks)

    payload = {"directory": directory}
    if config_overrides:
        payload["config_overrides"] = config_overrides
    if any_imgtype:
        payload["any_imgtype"] = True
    if dry_run:
        payload["dry_run"] = True
    if resubmit:
        payload["resubmit"] = True
    url = f"{host}/submit"
    if num_workers is not None:
        url += f"?num_workers={num_workers}"

    # urllib, not requests: `requests` is a third-party import this path does
    # not otherwise need, and the whole point here is to import as little as
    # possible. The body is small JSON and the response is small JSON.
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )

    log(f"submitting {directory} to {host} after {time.monotonic() - started:.1f}s")
    try:
        # Generous: this one call covers the server's resolve, prepare and the
        # whole fit, so it is bounded by the job rather than by the network.
        with urllib.request.urlopen(request, timeout=timeout + 300) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # The server puts its reason in the body as `detail`; urllib's own
        # message is only the status line, which on a 400 would hide which
        # directory was wrong and why.
        detail = exc.read().decode(errors="replace")[:500]
        if exc.code == 409:
            # Already submitted. Exit 0, like the trigger's own state-dir
            # guard: a hook firing twice is expected, and the second firing
            # declining is the guard working rather than a failure.
            log(f"already submitted: {detail}")
            return 0
        log(f"failed: HTTP {exc.code} {detail}")
        return 1
    except urllib.error.URLError as exc:
        log(f"failed: cannot reach {host}: {exc.reason}")
        return 1

    found = body.get("exposure") or {}
    state = body.get("state")
    log(
        f"visit={found.get('visit')} filter={found.get('physical_filter')} "
        f"imgtype={found.get('imgtype')} detectors={found.get('detectors')}"
    )
    if state == "DRY_RUN":
        log(
            f"dry run, would submit: boresight="
            f"({found.get('boresight_ra')}, {found.get('boresight_dec')}) "
            f"calib_time={found.get('calib_time')}"
        )
        return 0
    if body.get("note"):
        log(f"note: {body['note']}")
    if state == "SKIPPED":
        # 0, not 1: declining to fit a bias frame is correct behaviour, and the
        # camera's hook should not see a failure for it.
        log(f"skipped: {body.get('reason')}")
        return 0

    # Both timing blocks, flat, one per line. The resolving path used to print
    # these and losing them would make a slow exposure unattributable from the
    # hook's own log -- which is the only record a detached run leaves. `reused`
    # on the prepare side is the one to read first: a cold calib load is several
    # seconds and says nothing about the exposure.
    for label, timings in (
        ("prepare", body.get("prepare_timings")),
        ("submit", body.get("timings")),
    ):
        for key, value in sorted((timings or {}).items()):
            log(f"    {label} {key}: {value}")
    log(f"{state} in {time.monotonic() - started:.1f}s")
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
        "--resubmit",
        action="store_true",
        help="run an exposure the server has already accepted. The once-only "
        "guard is the server's and is keyed on the visit, so the same pixels "
        "under a different directory name are still caught",
    )
    parser.add_argument(
        "--any-imgtype",
        action="store_true",
        help=f"process even {'/'.join(sorted(protocol.SKIP_IMGTYPES))} frames, which are "
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
        # Poll, then the server's whole resolve/prepare/fit, then a wide
        # margin for everything unmodelled. No --wait to add any more:
        # /submit's own response is what bounds the fit.
        arm_watchdog(args.timeout + 600)

    if not args.token and not _host_is_loopback(args.host):
        parser.error(
            f"--token is required for a remote --host ({args.host}); "
            "or set DONUT_SERVER_TOKEN"
        )

    try:
        return run_notifier(
            args.directory,
            args.host,
            args.token,
            timeout=args.timeout,
            interval=args.poll_interval,
            stable_checks=args.stable_checks,
            config_overrides=args.config_overrides,
            num_workers=args.num_workers,
            any_imgtype=args.any_imgtype,
            dry_run=args.dry_run,
            resubmit=args.resubmit,
        )
    except TimeoutError as exc:
        # The poll gave up: the exposure never completed on disk. Reported as a
        # line rather than a traceback, since the hook reads a log.
        log(f"timed out: {exc}")
        return 1
