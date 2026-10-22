"""FastAPI front-end. Async/threaded -- never forks. All compute is delegated
to the coordinator process over a multiprocessing Pipe, serialized behind an
asyncio lock (Coord.send_command).

Raw pixels do NOT travel over that Pipe. A `multiprocessing.Pipe` is a
Unix-domain socketpair with an 8 KB buffer on macOS, which turns a ~308 MB push
into ~37,700 blocking round-trips (measured: 8.5 s, versus 0.16 s for the same
bytes over loopback HTTP). Instead the request body is streamed straight into a
reusable shared-memory block and only the part layout crosses the Pipe.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import itertools
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import shared_memory
from multiprocessing.connection import wait as mp_wait
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from lsst.ts.donut_server import logtail
from lsst.ts.donut_server import protocol
from lsst.ts.donut_server import table_view

MAX_PUSH_BYTES = 512 * 1024 * 1024  # early size-cap rejection

# The block also holds the header and descriptors, which sit ahead of the
# payload that MAX_PUSH_BYTES caps.
SHM_SIZE = MAX_PUSH_BYTES + 1024 * 1024

# Where a tmpfs-backed shm_open puts its blocks, when there is one. Checked for
# existence rather than assumed: this is the Linux layout, and on macOS --
# a supported dev platform -- POSIX shared memory is kernel-backed with no
# mount to interrogate at all, so the capacity check below is skipped there.
# The path is not configurable: CPython prefixes the block name with "/" and
# hands it to shm_open, which resolves it under this mount wherever it exists.
SHM_MOUNT = "/dev/shm"

# Caps on a prepare's -c/-C override list. Bounded rather than trusted because an
# accepted list is retained in _primed_args, replayed verbatim on every coordinator
# restart, and echoed (in digest form) from /health, which the dashboard polls at
# 1 Hz -- so an unbounded body would be a permanent cost, not a one-off one.
MAX_OVERRIDES = 64
MAX_OVERRIDE_TEXT_BYTES = 256 * 1024
MAX_OVERRIDE_TOTAL_BYTES = 1024 * 1024

# "value" is -c field=value; "python" is the contents of a -C file.
OVERRIDE_KINDS = ("value", "python")

# How long the reader thread sleeps between liveness checks. Both wakeup sources
# in _await_reply are load-bearing; see its docstring.
POLL_SLICE_S = 0.25

# Consecutive failed bring-ups before giving up and going DEGRADED.
MAX_RESTART_ATTEMPTS = 3
RESTART_BACKOFF_S = (0.0, 2.0, 5.0)

# The child spends ~15 s importing afw + ts_wep before it can say hello.
HELLO_TIMEOUT_S = 120.0

# How long a request will block waiting for a still-STARTING coordinator.
STARTUP_WAIT_S = 180.0

# How long a child gets to exit on its own after being told to shut down, before it
# is signalled. Polled rather than join()ed on a fixed timeout, so a healthy child
# costs only the finalization time it actually needs (measured: ~0.1 s) and a wedged
# one costs this bound. Generous because the coordinator finalizes the whole LSST
# stack here, and SIGTERMing it partway through that is how a shutdown path
# accumulates half-released OS resources.
GRACEFUL_EXIT_S = 5.0

# Reap budget once we have given up on a graceful exit: SIGTERM, then SIGKILL.
REAP_TERM_JOIN_S = 2.0
REAP_KILL_JOIN_S = 2.0

# Best-effort lock acquisition at shutdown, so no memoryview onto the shared
# block is live when it is closed and unlinked.
CLOSE_LOCK_WAIT_S = 2.0

# JobRecord.created_at is time.monotonic(), which has no defined epoch. Captured
# once here so the dashboard can render a wall clock, which it needs because the
# pipeline log's datefmt is "%H:%M:%S" (coordinator.py:613) -- time only, no date
# -- so lining a job up with its log lines takes a clock, not an age.
_BOOT_WALL = time.time() - time.monotonic()

# /proc/<pid>/statm counts pages, not bytes. Resolved once: it cannot change.
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DASHBOARD_HTML = os.path.join(_STATIC, "dashboard.html")
RESULTS_HTML = os.path.join(_STATIC, "results.html")

# JOBS is unbounded and each DONE record holds ~200 KB of parquet, so the job
# list is paged. Newest first, since that is what a dashboard reads.
JOBS_PAGE_DEFAULT = 50
JOBS_PAGE_MAX = 500


class JobState(str, Enum):
    PREPARED = "PREPARED"
    RECEIVING = "RECEIVING"
    COMPUTING = "COMPUTING"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass
class JobRecord:
    job_id: str
    state: JobState
    created_at: float = field(default_factory=time.monotonic)
    result: Any = None
    error: Optional[str] = None
    prepare_timings: Optional[dict] = None
    push_timings: Optional[dict] = None
    # Serialized donutBlitzCornerResults, served by /result/{job_id}/table. Held
    # opaque bytes: this process never imports astropy or the LSST stack.
    table_parquet: Optional[bytes] = None
    # The coordinator's composite cache key for the config this job was prepared
    # against. Opaque here -- never inspected, just replayed on push so the
    # coordinator reactivates (or, if evicted, reloads) *this* job's config
    # rather than whatever the most recent prepare happened to leave active.
    prepared_key: Optional[list] = None


JOBS: dict[str, JobRecord] = {}


def _stamp_path(job_id: str) -> Optional[str]:
    """Where the coordinator's image table for this job would be, or None.

    Reads the environment directly rather than calling coordinator.stamp_dir(),
    which looks like the obvious reuse but is not available here: importing
    coordinator pulls the whole LSST stack into the front-end (778 MB, see
    _resolve_target). The shared contract is the variable, not the function.

    None for unset, not a raise: the coordinator already refuses to prepare a job
    without this set, so by the time any job exists it is configured. Erroring
    here would only convert an operator's misconfiguration into a 500 on a job
    that legitimately has no images.
    """
    directory = os.environ.get("DONUT_SERVER_STAMP_DIR")
    if not directory:
        return None
    return os.path.join(directory, f"{job_id}.parquet")


def _plot_path(job_id: str, name: str, size: str) -> Optional[str]:
    """Where one rendered PNG would be, or None when the output dir is unset.

    Mirrors coordinator.plot_dir, and reads the environment directly for the same
    reason _stamp_path does: the shared contract is the variable, not the
    function, because importing coordinator would pull 778 MB of LSST stack into
    this process.

    `name` and `size` are **not** sanitized here -- every caller must have already
    checked them against PLOT_NAMES/PLOT_SIZES, which is what keeps a URL segment
    from reaching the filesystem as a path.
    """
    directory = os.environ.get("DONUT_SERVER_STAMP_DIR")
    if not directory:
        return None
    return os.path.join(directory, "plots", job_id, f"{name}.{size}.png")


@dataclass
class ConfigSnapshot:
    """The blitz task's full config as of the last successful prepare.

    Cached here rather than fetched on demand for two reasons. The coordinator's
    command loop is serial and has no command timeout by design, so a `cmd: "config"`
    round-trip would queue behind an in-flight 6-9 s push plus its deferred stamp
    write -- turning a metadata GET into a multi-second block. And a cache still
    answers while the coordinator is RESTARTING or DEGRADED, which is exactly when an
    operator most wants to know what was running.

    Current by construction: the config can only change at prepare, and the
    coordinator returns the dump on every prepare reply including reused ones.
    """

    dump: str
    generation: int
    overrides: list  # digest form; never carries -C bodies
    at: float
    sha256: str


class CoordState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    RESTARTING = "restarting"
    DEGRADED = "degraded"
    STOPPING = "stopping"


class CoordinatorLost(RuntimeError):
    """The child died while we were talking to it."""


class CoordinatorUnavailable(RuntimeError):
    """No usable child right now (starting / restarting / degraded)."""


def _signal_child(pid: int, sig: int) -> None:
    """Signal the child's whole process group, falling back to the bare pid.

    coordinator_main calls os.setpgid(0, 0), so the child leads its own group and
    killpg reaches the task's fork workers and nothing else. killpg is tried
    *first*, and unconditionally: it can only ever deliver to the group whose id
    equals `pid`, and the front-end's own group id is its own pgid, which is never
    a child's pid -- so this cannot take the front-end down even if that setpgid
    failed and the child is sharing our group.

    Trying the group first also matters after the coordinator itself has been
    reaped: os.kill(pid) is then a no-op, but the group may still hold surviving
    fork workers, which would otherwise keep the shared-block mapping and 8 cores.
    """
    try:
        os.killpg(pid, sig)
        return
    except OSError:
        pass  # no such group: setpgid must have failed, so fall back to the pid
    try:
        os.kill(pid, sig)
    except OSError:
        pass  # already gone


class Coord:
    """Owns the coordinator process, the Pipe, and the shared raw-image block.

    Also owns coordinator liveness: it detects a dead child promptly (rather than
    blocking forever in recv()), restarts it in the background a bounded number of
    times, replays the last prepare, and otherwise settles into a terminal
    DEGRADED state that stays up and explains itself via /health.
    """

    def __init__(
        self,
        *,
        target: Optional[Callable] = None,
        target_args: tuple = (),
        shm_size: int = SHM_SIZE,
        max_restart_attempts: int = MAX_RESTART_ATTEMPTS,
        backoff: tuple = RESTART_BACKOFF_S,
        on_restart: Optional[Callable[[dict], None]] = None,
        on_fatal: Optional[Callable[[str], None]] = None,
        index: int = 0,
    ) -> None:
        # Which flight this is. Carried so a raised CoordinatorLost can name its
        # own lane without the handler having to search the pool for it.
        self.index = index
        self._proc = None
        self._conn = None
        self._shm = None
        self._lock = asyncio.Lock()
        # The shared block is a single reusable buffer, so only one push may be
        # writing into it at a time. One-job-at-a-time is already the design,
        # but until now only *coordinator* access was serialized; the body
        # receive was not.
        self.push_lock = asyncio.Lock()

        # Seams for tests, all defaulting to production behaviour. `target` must
        # be a module-level function: spawn pickles it by reference, so a fake
        # target's module is imported in the child and the LSST stack is never
        # touched.
        self._target = target
        self._target_args = tuple(target_args)
        self._shm_size = shm_size
        self._max_restart_attempts = max_restart_attempts
        self._backoff = tuple(backoff)
        self._on_restart = on_restart
        # Called instead of retrying when the child reports a fatal config error.
        # A seam rather than a direct _shutdown_server() call so the lifecycle
        # tests can observe the decision without taking pytest down with it.
        self._on_fatal = on_fatal

        # A *dedicated* executor, not the default one: uvicorn's shutdown calls
        # loop.shutdown_default_executor(THREAD_JOIN_TIMEOUT) with
        # THREAD_JOIN_TIMEOUT = 300, so a reader thread parked in the default
        # executor would make SIGTERM take five minutes.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coord-io")
        self._abort = threading.Event()

        self._state = CoordState.STARTING
        # Set once STARTING has resolved -- to READY *or* to DEGRADED. Waiters
        # must re-check _state after waking.
        self._ready_event = asyncio.Event()
        self._restart_task: Optional[asyncio.Task] = None
        self._startup_task: Optional[asyncio.Task] = None
        self._in_flight = 0

        self._generation = 0
        self._spawns = 0
        self._restart_attempts = 0
        # Completed work on this lane, counted for the life of the *front-end*
        # rather than of the current child: it survives a restart on purpose,
        # because "this flight has served 47 exposures" is the throughput
        # question, and a counter that reset on every respawn would answer a
        # different one. See note_exposure.
        self._exposures = 0
        self._child_pid: Optional[int] = None
        self._child_started_at: Optional[float] = None
        self._last_loss: Optional[dict] = None
        self._degraded_reason: Optional[str] = None
        self._primed_args: Optional[dict] = None
        self._loss_provoked_by: Optional[str] = None
        self._reprime_error: Optional[str] = None
        # Set when the child reported a non-retryable config error. Distinct from
        # _degraded_reason, which means "gave up after N attempts": this one was
        # never retried, and /health should say which happened.
        self._fatal_reason: Optional[str] = None
        # One slot, never per-job: the dump is ~77 KB and must not reach a JobRecord.
        self._config_snapshot: Optional[ConfigSnapshot] = None

    # ---------------------------------------------------------------- accessors

    @property
    def shm(self):
        return self._shm

    @property
    def shm_size(self) -> int:
        return self._shm_size

    @property
    def state(self) -> CoordState:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def spawns(self) -> int:
        return self._spawns

    @property
    def restart_attempts(self) -> int:
        return self._restart_attempts

    @property
    def exposures(self) -> int:
        """Exposures this flight has computed since the front-end started."""
        return self._exposures

    def note_exposure(self) -> None:
        """Count one exposure as served by this flight.

        Called from the push path on success only, and deliberately not derived
        from JOBS: that is paged and, more importantly, the dashboard's `limit`
        bounds what it can see, so counting DONE rows there would report "how
        many of the newest 50 jobs ran here" rather than a total. Kept on the
        Coord because the number is per-lane, which is the comparison the
        side-by-side flight panels exist to support.
        """
        self._exposures += 1

    @property
    def child_pid(self) -> Optional[int]:
        return self._child_pid

    @property
    def last_loss(self) -> Optional[dict]:
        return self._last_loss

    @property
    def degraded_reason(self) -> Optional[str]:
        return self._degraded_reason

    @property
    def fatal_reason(self) -> Optional[str]:
        """Set only when bring-up hit a config error it refused to retry.

        Separate from degraded_reason so /health distinguishes "this service is
        misconfigured" from "the child keeps dying": the first wants an operator
        editing a command line, the second wants a look at the traceback.
        """
        return self._fatal_reason

    @property
    def primed_args(self) -> Optional[dict]:
        return self._primed_args

    @property
    def config_snapshot(self) -> Optional[ConfigSnapshot]:
        return self._config_snapshot

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def busy(self) -> bool:
        """Whether a command is running or queued on this flight.

        Both terms are needed: _lock is held for the whole of send_command, which
        includes the _ensure_usable wait before anything is in flight, and
        _in_flight is non-zero during bring-up's hello, which does not hold _lock.
        """
        return self._in_flight > 0 or self._lock.locked()

    @property
    def child_uptime_s(self) -> Optional[float]:
        if self._child_started_at is None:
            return None
        return time.monotonic() - self._child_started_at

    def is_alive(self) -> bool:
        """Liveness only. Deliberately NOT readiness: this is True for a child 15 s
        into importing afw + ts_wep, and for a child whose fork pool is wedged."""
        return self._proc is not None and self._proc.is_alive()

    # ---------------------------------------------------------------- detection

    def _await_reply(self, conn, proc, timeout: Optional[float] = None) -> dict:
        """Wait for a reply, or for the child to die. Runs on the reader thread.

        `proc.is_alive()` -- waitpid(WNOHANG) on the coordinator itself -- is the
        load-bearing source, and the only one immune to descendants. Measured: when
        the coordinator dies with even one fork worker still up, *neither* fd-based
        source reports it. popen_fork closes only the fds it creates, so the task's
        8 workers inherit copies of both the spawn sentinel's write end and the pipe
        itself; the sentinel stays unreadable and the pipe never reaches EOF, so a
        plain recv() blocks indefinitely. That is the permanent wedge, and it is
        exactly the crash-during-runQuantum case -- the likeliest crash, since that
        is where all the compute and all the 4-6 GB RSS oscillation lives.

        The pipe is still polled alongside it because that is how replies arrive,
        and the sentinel because it is prompt in the no-descendants case.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._abort.is_set():
                raise CoordinatorLost("shutting down")
            if conn in mp_wait([conn, proc.sentinel], timeout=POLL_SLICE_S):
                try:
                    return conn.recv()
                except (EOFError, OSError) as exc:  # includes a truncated message
                    # EOFError stringifies to "", so name the type: this text is
                    # what the producer sees in the 503 body and in last_loss.
                    detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                    raise CoordinatorLost(f"pipe closed ({detail})") from exc
            if not proc.is_alive():
                raise CoordinatorLost(f"coordinator exited (exitcode={proc.exitcode!r})")
            if deadline is not None and time.monotonic() >= deadline:
                raise CoordinatorLost(f"no reply within {timeout:.0f}s")

    async def _run_reader(self, fn, *args) -> dict:
        loop = asyncio.get_running_loop()
        self._in_flight += 1
        try:
            return await loop.run_in_executor(self._executor, fn, *args)
        finally:
            self._in_flight -= 1

    async def _exchange(self, command: dict) -> dict:
        """One send/recv round-trip. Caller must hold _lock."""
        proc, conn = self._proc, self._conn
        if proc is None or conn is None:
            raise CoordinatorUnavailable("no coordinator process")

        def _do():
            try:
                conn.send(command)
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise CoordinatorLost(f"send failed: {exc}") from exc
            return self._await_reply(conn, proc)

        # Deliberately NOT wrapped in asyncio.wait_for. Cancelling the await does
        # not stop the thread: _lock would be released while a live thread still
        # owns self._conn, and a late reply could then be handed to the *next* job.
        return await self._run_reader(_do)

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        # Created once and reused for every job: no per-job create/unlink, and
        # therefore no way to leak a block per job. Pages are faulted in lazily,
        # so RSS tracks the largest push actually received, not shm_size.
        self._shm = shared_memory.SharedMemory(create=True, size=self._shm_size)
        self._state = CoordState.STARTING
        # Bring-up runs in the background so /health can answer "starting"
        # accurately during the child's ~15 s of imports. It deliberately does not
        # hold _lock: requests are gated on _state / _ready_event instead.
        self._startup_task = asyncio.create_task(self._bring_up())

    def _resolve_target(self):
        if self._target is not None:
            return self._target
        # This DOES load the LSST stack (afw, ip_isr, daf_butler, ts_wep) into
        # the front-end -- measured at 778 MB RSS, with libopenblas and libomp
        # mapped in. Spawn pickles the target by reference, so the child imports
        # `coordinator` again itself; deferring the import to here rather than
        # module scope only moves when the front-end pays for it, it does not
        # avoid it.
        from lsst.ts.donut_server import coordinator

        self._target = coordinator.coordinator_main
        return self._target

    def _spawn_child(self) -> None:
        target = self._resolve_target()
        # Spawn, never fork: this process is async/threaded.
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = mp.Pipe()
        # The block's name is passed only when there is one. A plot flight has no
        # shared block -- raw pixels never reach it -- so its target takes the
        # connection alone; see PlotFlight.
        block = () if self._shm is None else (self._shm.name,)
        self._proc = ctx.Process(
            target=target, args=(child_conn,) + block + self._target_args
        )
        self._proc.start()
        # Drop the parent's copy of the child end: a socketpair reaches EOF only
        # when every copy of the peer fd is closed. Explicit rather than load-
        # bearing -- measured, CPython refcounting already closed this the moment
        # the enclosing frame went away (Process.start() deletes its own reference
        # to args), so EOF was reachable without it. Kept because it makes the
        # intent local and survives a refactor that retains the reference, but the
        # wedge this service actually suffers from is fd inheritance by the task's
        # fork workers, which no amount of closing here can fix; see _await_reply.
        child_conn.close()
        self._conn = parent_conn
        self._spawns += 1
        self._child_pid = self._proc.pid

    def _reap(self) -> None:
        """Take the current child fully down and release its fds. Blocking."""
        proc, conn = self._proc, self._conn
        self._proc = None
        self._conn = None
        self._child_started_at = None
        if proc is None:
            if conn is not None:
                conn.close()
            return

        pid = proc.pid
        # Kill the *group*, not the process: orphaned fork workers would otherwise
        # survive holding the shared-block mapping and 8 cores. Signalled even when
        # the coordinator itself is already dead, because that is exactly when the
        # stragglers it forked are left without a parent to clean them up.
        if pid is not None:
            _signal_child(pid, signal.SIGTERM)
            proc.join(timeout=REAP_TERM_JOIN_S)
            if proc.is_alive():
                _signal_child(pid, signal.SIGKILL)
                proc.join(timeout=REAP_KILL_JOIN_S)
        try:
            # Releases the sentinel fd. Without it every restart leaks one.
            proc.close()
        except ValueError:
            pass
        if conn is not None:
            conn.close()

    async def _bring_up(self) -> None:
        """Spawn, handshake, re-prime -- retrying up to max_restart_attempts.

        On exhaustion: DEGRADED, which is terminal until POST /admin/restart.
        """
        loop = asyncio.get_running_loop()
        while True:
            if self._state is CoordState.STOPPING:
                return
            if self._restart_attempts >= self._max_restart_attempts:
                last = (self._last_loss or {}).get("reason", "unknown")
                self._degraded_reason = (
                    f"{self._restart_attempts} consecutive bring-up attempts failed; "
                    f"last: {last}"
                )
                self._state = CoordState.DEGRADED
                # Wake anyone blocked waiting for startup to resolve.
                self._ready_event.set()
                return

            delay = self._backoff[min(self._restart_attempts, len(self._backoff) - 1)]
            self._restart_attempts += 1
            if delay:
                await asyncio.sleep(delay)

            await loop.run_in_executor(self._executor, self._reap)
            try:
                self._spawn_child()
                msg = await self._run_reader(
                    self._await_reply, self._conn, self._proc, HELLO_TIMEOUT_S
                )
                if msg.get("event") == "fatal":
                    # The child found the service misconfigured -- a repo that
                    # does not open, collections that name nothing. Retrying
                    # cannot help: every replacement child reads the same
                    # environment and would reach the same conclusion, so the
                    # backoff loop would only delay the report by ~7 s and then
                    # land in DEGRADED, which looks like a crash loop rather
                    # than a typo. Report it and take the server down.
                    reason = msg.get("error", "calib configuration is unusable")
                    self._fatal_reason = reason
                    self._degraded_reason = reason
                    self._state = CoordState.DEGRADED
                    await loop.run_in_executor(self._executor, self._reap)
                    self._ready_event.set()
                    if self._on_fatal is not None:
                        self._on_fatal(reason)
                    return
                if msg.get("event") != "hello":
                    raise CoordinatorLost(f"expected hello, got {msg!r}")
            except CoordinatorLost as exc:
                self._note_loss(exc, provoked_by=self._loss_provoked_by)
                continue

            self._generation += 1
            self._child_pid = msg.get("pid", self._proc.pid)
            self._child_started_at = time.monotonic()
            # A successful hello resets the counter, so unrelated crashes hours
            # apart never accumulate into DEGRADED.
            self._restart_attempts = 0

            try:
                await self._reprime()
            except CoordinatorLost as exc:
                self._note_loss(exc, provoked_by="prepare")
                continue

            self._loss_provoked_by = None
            self._state = CoordState.READY
            self._ready_event.set()
            return

    def _note_prepared(self, command: dict, resp: dict) -> None:
        """Record what a successful prepare installed on the child.

        Called from both places a prepare can succeed -- send_command and _reprime.
        That second call site is the reason this is a method rather than two lines
        inline: _reprime talks to the child via _exchange directly, bypassing
        send_command entirely, so anything cached only in send_command silently would
        not be refreshed after a restart. Harmless while _primed_args was the only
        thing at stake (the replayed dict is identical), but the snapshot carries a
        generation, and a stale one would misreport which child's config is live.
        """
        self._primed_args = dict(command)
        dump = resp.get("config_dump")
        if dump is None:
            return
        self._config_snapshot = ConfigSnapshot(
            dump=dump,
            generation=self._generation,
            overrides=_digest_overrides(command.get("config_overrides")),
            at=time.time(),
            sha256=hashlib.sha256(dump.encode()).hexdigest(),
        )

    async def _reprime(self) -> None:
        """Replay the last successful prepare, so the producer does not have to.

        Skipped when the loss was provoked *by* a prepare: replaying the command
        that killed the last child would very likely kill this one too.
        """
        args = self._primed_args
        if args is None or self._loss_provoked_by == "prepare":
            return
        self._reprime_error = None
        try:
            resp = await self._exchange(dict(args))
        except CoordinatorUnavailable as exc:
            self._reprime_error = str(exc)
            return
        if resp.get("ok"):
            # Refreshes the snapshot's generation to this new child.
            self._note_prepared(args, resp)
        else:
            # Auto-re-priming is an availability optimization, not a correctness
            # requirement: run_job cross-checks the raws' physical_filter against
            # the loaded calibs, and the refcat coverage check is equivalent, so an unprimed
            # coordinator fails the next push loudly rather than producing bad
            # wavefronts. Leave the child READY but unprimed.
            self._primed_args = None
            self._reprime_error = resp.get("error", "prepare replay failed")

    def _note_loss(self, exc: Exception, provoked_by: Optional[str] = None) -> None:
        proc = self._proc
        # proc.exitcode is populated by waitpid, which only happens inside
        # join()/is_alive(). A pipe-EOF loss reaches here without ever having
        # called either, so exitcode can still read as the pre-reap None even
        # though the child has already exited (EOF on the pipe implies the
        # child is gone or about to be). Short timeout: the process is already
        # dead or dying, so this should return almost immediately.
        if proc is not None:
            proc.join(timeout=1.0)
        self._last_loss = {
            "reason": str(exc),
            "exitcode": proc.exitcode if proc is not None else None,
            "pid": self._child_pid,
            "provoked_by": provoked_by,
            "generation": self._generation,
            "at": time.time(),
        }
        self._loss_provoked_by = provoked_by
        self._child_started_at = None
        if self._on_restart is not None:
            self._on_restart(self._last_loss)

    def _start_restart(self) -> None:
        """Move to RESTARTING and kick off exactly one background restart."""
        if self._state is CoordState.STOPPING:
            return
        self._state = CoordState.RESTARTING
        self._ready_event.clear()
        if self._restart_task is None:
            self._restart_task = asyncio.create_task(self._restart())

    async def _restart(self) -> None:
        """Reap and replace the child. Runs in the background rather than inside
        the failing request: reap + spawn + hello + re-prime is ~19 s, and making
        the client wait that long for a 503 is worse than answering immediately."""
        try:
            async with self._lock:
                await self._bring_up()
        finally:
            self._restart_task = None

    def kill_child(self) -> Optional[int]:
        """Force the child down from outside _lock -- the answer to
        wedged-but-alive, where is_alive() is True but nothing will ever reply.

        Only signals; it never closes a Connection, which would race _reap into a
        use-after-close. A blocked _await_reply notices the death via
        proc.is_alive() within POLL_SLICE_S and raises CoordinatorLost, so the
        ordinary restart path runs on the coroutine that already holds _lock.
        """
        proc = self._proc
        if proc is None or not proc.is_alive() or proc.pid is None:
            return None
        _signal_child(proc.pid, signal.SIGKILL)
        return proc.pid

    def request_restart(self, reason: str) -> Optional[int]:
        """Operator-driven restart. Safe to call in any state."""
        if self._state is CoordState.DEGRADED:
            # Nothing is in flight in DEGRADED, and there is no child to kill.
            self._restart_attempts = 0
            self._degraded_reason = None
            # Cleared too, so a retry after the operator has fixed the
            # environment is not reported as still-fatal. It cannot *succeed*
            # without a fix -- the child re-reads the same environment, finds it
            # still bad, and sets this again -- which is the point: the flag
            # tracks the last bring-up, not a permanent verdict.
            self._fatal_reason = None
            self._start_restart()
            return None

        pid = self.kill_child()
        if self._in_flight == 0:
            # No reader is parked on the pipe, so nothing would notice the death.
            self._note_loss(RuntimeError(reason), provoked_by=None)
            self._start_restart()
        return pid

    async def _ensure_usable(self) -> None:
        """Caller holds _lock. Raises unless there is a child ready to serve."""
        if self._state is CoordState.READY:
            return
        if self._state is not CoordState.STARTING:
            if self._state is CoordState.DEGRADED:
                raise CoordinatorUnavailable(self._degraded_reason or "coordinator degraded")
            raise CoordinatorUnavailable(f"coordinator {self._state.value}")

        # STARTING waits; RESTARTING (above) 503s immediately. Asymmetric on
        # purpose: startup latency is not an anomaly, and blocking here preserves
        # the pre-existing behaviour -- whereas a restart *is* an anomaly the
        # producer should hear about promptly, at a 30 s job cadence.
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=STARTUP_WAIT_S)
        except asyncio.TimeoutError:
            raise CoordinatorUnavailable("coordinator still starting") from None
        if self._state is not CoordState.READY:
            if self._state is CoordState.DEGRADED:
                raise CoordinatorUnavailable(self._degraded_reason or "coordinator degraded")
            raise CoordinatorUnavailable(f"coordinator {self._state.value}")

    async def send_command(self, command: dict) -> dict:
        async with self._lock:
            try:
                await self._ensure_usable()
            except CoordinatorUnavailable as exc:
                # Tagged so the 503 handler reports *this* flight's state rather
                # than the pool's aggregate, which may be healthy.
                exc.flight = self.index  # type: ignore[attr-defined]
                raise
            try:
                resp = await self._exchange(command)
            except CoordinatorLost as exc:
                self._note_loss(exc, provoked_by=command.get("cmd"))
                # Set under _lock, because requests arriving mid-restart must be
                # able to check state *without* the lock -- _restart holds it.
                self._start_restart()
                exc.flight = self.index  # type: ignore[attr-defined]
                # The failed command is deliberately not retried: a push that
                # killed the coordinator may kill it again, and the job's state
                # machine has already moved on. 503 and let the producer decide.
                raise
            if command.get("cmd") == "prepare" and resp.get("ok"):
                self._note_prepared(command, resp)
            return resp

    async def aclose(self) -> None:
        self._state = CoordState.STOPPING
        self._abort.set()
        self._ready_event.set()

        for task in (self._restart_task, self._startup_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Lock order is push_lock then _lock, as in push(). push_lock matters most:
        # a live memoryview onto the block makes close()/unlink() raise BufferError.
        held = []
        for lock in (self.push_lock, self._lock):
            try:
                await asyncio.wait_for(lock.acquire(), timeout=CLOSE_LOCK_WAIT_S)
                held.append(lock)
            except asyncio.TimeoutError:
                pass  # best effort; tear the child down regardless
        try:
            conn, proc = self._conn, self._proc
            if conn is not None and proc is not None and proc.is_alive():
                try:
                    conn.send({"cmd": "shutdown"})
                except OSError:
                    pass
                else:
                    # Polled rather than join()ed on a fixed timeout, so a healthy
                    # child costs only the time it actually needs to finalize and a
                    # wedged one costs at most GRACEFUL_EXIT_S. await, not sleep:
                    # this must not block the loop uvicorn is still shutting down.
                    deadline = time.monotonic() + GRACEFUL_EXIT_S
                    while proc.is_alive() and time.monotonic() < deadline:
                        await asyncio.sleep(0.05)
            self._reap()
            self._executor.shutdown(wait=False)
            if self._shm is not None:
                # This process created the block, so it is the one that unlinks it.
                try:
                    self._shm.close()
                    self._shm.unlink()
                except BufferError:
                    pass  # a view is still exported; the OS reclaims it at exit
                self._shm = None
        finally:
            for lock in held:
                lock.release()


# Set by the first flight to report a fatal config error, so the N-1 others
# reporting the same thing neither duplicate the banner nor re-signal.
_FATAL_REPORTED = threading.Event()


def _fatal_config_handler(reason: str) -> None:
    """Report a misconfiguration to whoever started the server, then stop it.

    Printed to stderr, not logged: the audience is the operator watching the
    terminal they just typed the command into, and this process configures no
    logging handler at all (the coordinator child owns the pipeline log), so a
    _log call here would reach the same stderr via logging.lastResort and say
    it twice.

    SIGTERM to our own process rather than sys.exit or os._exit: this runs on
    the event loop, where sys.exit only unwinds one task, and os._exit would
    skip uvicorn's lifespan shutdown -- which is what reaps the other flights'
    children and unlinks their shared blocks. SIGTERM is the signal uvicorn
    already installs a graceful handler for, so the existing shutdown path runs
    exactly as it does for a Ctrl-C.
    """
    if _FATAL_REPORTED.is_set():
        return
    _FATAL_REPORTED.set()
    print(f"\nFATAL: {reason}", file=sys.stderr)
    print(
        "The server cannot serve any request in this state; shutting down.",
        file=sys.stderr,
        flush=True,
    )
    os.kill(os.getpid(), signal.SIGTERM)


class ShmTooSmall(RuntimeError):
    """`SHM_MOUNT` cannot supply the shared blocks the flights will need.

    Its own type, and raised rather than reported through `_fatal_config_handler`:
    this happens inside the lifespan startup, before any flight has a child to
    lose, so there is nothing yet to shut down gracefully -- letting it propagate
    makes uvicorn abort startup and print it, which is the same operator-facing
    outcome with none of the signal plumbing.
    """


def shm_free_bytes(mount: str = SHM_MOUNT) -> int | None:
    """Bytes available on `mount`, or None if there is nothing to measure.

    None covers both "no such mount" (macOS, where POSIX shared memory is
    kernel-backed) and a statvfs that fails for any reason: in neither case do we
    know the capacity, and a guess would either refuse a working server or
    promise capacity that is not there.
    """
    try:
        st = os.statvfs(mount)
    except OSError:
        return None
    # f_bavail, not f_bfree: the latter counts root-only reserve a server running
    # as an ordinary user cannot have. For tmpfs they are equal, but this is also
    # the correct read if SHM_MOUNT is ever something else.
    return st.f_bavail * st.f_frsize


def rss_bytes(pid: Optional[int]) -> Optional[int]:
    """Resident set size of `pid`, or None if it cannot be measured.

    Read straight out of /proc rather than through psutil, which is not a
    dependency of this package: one open-and-split of a two-field file, cheap
    enough for a 1 Hz /health.

    Deliberately *this process only*, not its process group. The blitz task forks
    eight cutout workers per push whose pages are overwhelmingly copy-on-write
    shares of the coordinator's own 4-6 GB, so a group sum would report 30+ GB of
    memory that does not exist. The parent's RSS is both the honest number and the
    one that moves when a filter's calibs land.

    None covers a dead or never-spawned child, a pid that exited between the
    /health handler reading it and this call, and any platform without /proc.
    """
    if pid is None:
        return None
    try:
        with open(f"/proc/{pid}/statm", "rb") as f:
            resident = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident * _PAGE_SIZE


def check_shm_capacity(n_flights: int, shm_size: int = SHM_SIZE) -> None:
    """Fail startup if the shared blocks cannot be backed by real pages.

    The failure this prevents is otherwise undiagnosable. `SharedMemory(create=
    True)` only does shm_open + ftruncate, which on tmpfs reserves nothing and
    succeeds however small the mount is; the mmap succeeds too. The shortfall is
    discovered only when /push *writes* a page tmpfs cannot allocate, and the
    kernel's sole recourse at that point is SIGBUS -- which kills the front-end
    mid-request, so the producer sees a closed socket with no status and the
    server leaves nothing behind but "Bus error (core dumped)".

    Checked across the whole pool rather than per flight for the same reason the
    runtime failure is late: a sparse block does not reduce f_bavail, so flight 0
    reserving 513 MB leaves flight 1 measuring the same free space and finding it
    ample. Only the sum is a real requirement.

    A floor, not a reservation: nothing stops another process on the host taking
    the space between here and the first push. It catches the standing
    misconfiguration -- a 64 MB container default against a 513 MB block -- which
    is the case that can never work rather than the one that got unlucky.
    """
    free = shm_free_bytes()
    if free is None:
        return
    needed = n_flights * shm_size
    if free >= needed:
        return
    mb = 1024 * 1024
    raise ShmTooSmall(
        f"{SHM_MOUNT} has {free / mb:.0f} MB available but this server needs "
        f"{needed / mb:.0f} MB ({n_flights} "
        f"flight{'s' if n_flights != 1 else ''} x {shm_size / mb:.0f} MB). "
        "Each flight streams one push into a shared block of that size, and a "
        "block short of pages kills the server with SIGBUS mid-push rather than "
        f"failing the request. Give {SHM_MOUNT} more room (a container usually "
        "defaults to 64 MB; --shm-size=2g or more), or run fewer flights with "
        "--num-flights."
    )


class FlightPool:
    """N independent coordinator processes ("flights"), and the routing between
    them and the jobs they prepared.

    Concurrency comes entirely from there being N `Coord`s: each already owns its
    own child, Pipe, shared block, command lock and push lock, so two jobs in
    COMPUTING at once needs no new mechanism -- only a decision about which flight
    a given job belongs to, which is what this class is.

    Nothing here is shared between flights. In particular a flight's crash must
    not touch a job that a different flight is happily computing, which is why
    `_fail_jobs` filters on the routing table rather than failing everything
    mid-flight.
    """

    def __init__(self, n: int = 1, **coord_kwargs) -> None:
        # Popped before the comprehension, not inside it: a pop per iteration
        # would hand the caller's handler to flight 0 and the default to the rest.
        on_fatal = coord_kwargs.pop("on_fatal", None) or _fatal_config_handler
        self.flights = [
            Coord(
                index=i,
                on_restart=self._loss_handler(i),
                on_fatal=on_fatal,
                **coord_kwargs,
            )
            for i in range(n)
        ]
        # job_id -> flight index, written at /prepare and read at /push. Unbounded,
        # like JOBS itself: two small ints per job. Deliberately not reaped on its
        # own schedule -- a retention policy here that disagreed with JOBS' would
        # turn a still-listed job into an unroutable push.
        self._by_job: dict[str, int] = {}
        self._last_used = [0.0] * n
        # Prepares and pushes waiting on each flight's locks. Counted rather than
        # read off asyncio.Lock, whose waiter list is private.
        self._queued = [0] * n

    def __len__(self) -> int:
        return len(self.flights)

    def __getitem__(self, index: int) -> Coord:
        return self.flights[index]

    def _loss_handler(self, index: int) -> Callable[[dict], None]:
        def handler(loss: dict) -> None:
            self._fail_jobs(index, loss)

        return handler

    def _fail_jobs(self, index: int, loss: dict) -> None:
        """Fail the jobs that were mid-flight *on this flight* when its child died.

        Without this a record stranded in COMPUTING makes /result?wait=N long-poll
        a job that can never finish. Scoped to one flight because the alternative
        -- failing every non-terminal job, which is what a single-coordinator
        server could safely do -- would let one lane's crash kill another lane's
        healthy job for no reason the producer could ever diagnose.
        """
        for job_id, record in JOBS.items():
            if self._by_job.get(job_id) != index:
                continue
            if record.state in (JobState.RECEIVING, JobState.COMPUTING):
                was = record.state
                record.state = JobState.ERROR
                record.error = f"coordinator lost during {was.value}: {loss.get('reason')}"

    # ------------------------------------------------------------------ routing

    def select(self, command: dict) -> Coord:
        """Which flight should serve this prepare.

        In priority order, all non-blocking:

        1. A flight already primed with exactly this command. Free, and it keeps a
           repeated pointing on one lane rather than warming a second one.
        2. The least recently used flight that is READY and idle.
        3. The least recently used flight, whose lock the caller will then wait on.

        LRU rather than first-fit in 2, which is the difference between using the
        pool and not: prepares arrive serially, one having finished before the next
        is sent, so "the first idle flight" would be flight 0 every single time and
        no job would ever be routed anywhere else. LRU rather than round-robin
        because it is the flight holding the *least* useful config that should be
        retargeted.

        Only 3 can block, and it is also the only branch that can reach a
        non-READY flight -- which is deliberate: that is where `_ensure_usable`
        produces the ordinary 503, including when every flight is DEGRADED.
        """
        ready = [c for c in self.flights if c.state is CoordState.READY]
        for coord in ready:
            if coord.primed_args == command:
                return coord
        idle = [c for c in ready if not c.busy]
        return self._lru(idle or self.flights)

    def _lru(self, coords: list[Coord]) -> Coord:
        return min(coords, key=lambda c: self._last_used[c.index])

    def note_used(self, coord: Coord) -> None:
        self._last_used[coord.index] = time.monotonic()

    def route(self, job_id: str, coord: Coord) -> None:
        self._by_job[job_id] = coord.index
        self.note_used(coord)

    def for_push(self, job_id: str) -> Coord:
        """The flight that prepared `job_id`, or the LRU one if we have no record.

        The fallback is not a guess: the push carries the job's `prepared_key`, and
        the coordinator reloads that config from its own `_PREPARE_COMMANDS` before
        running, so any flight can serve any job -- just more slowly. Reachable when
        the routing entry predates a restart of the front-end but the job record
        does not.
        """
        index = self._by_job.get(job_id)
        if index is None:
            coord = self._lru(self.flights)
            self._by_job[job_id] = coord.index
            return coord
        return self.flights[index]

    def flight_of(self, job_id: str) -> Optional[int]:
        """Which flight prepared `job_id`, or None if nothing routed it.

        Unlike `for_push` this is a read: it never invents a route for a job that
        has none, because the dashboard asking "whose job is this" must not change
        where a later push goes.
        """
        return self._by_job.get(job_id)

    def queue_depth(self, coord: Coord) -> int:
        return self._queued[coord.index]

    @asynccontextmanager
    async def queued(self, coord: Coord):
        """Count a caller as backlog for the whole time it may be waiting."""
        self._queued[coord.index] += 1
        try:
            yield coord
        finally:
            self._queued[coord.index] -= 1

    # ---------------------------------------------------------------- aggregate

    @property
    def state(self) -> CoordState:
        """One verdict for the pool: READY if any flight is, else the worst state.

        Used by the 503 handler when the failure cannot be attributed to a single
        flight. Worst-first ordering so a pool with one DEGRADED flight and one
        still STARTING reports the degradation.
        """
        states = [c.state for c in self.flights]
        if CoordState.READY in states:
            return CoordState.READY
        for state in (CoordState.DEGRADED, CoordState.STOPPING,
                      CoordState.RESTARTING, CoordState.STARTING):
            if state in states:
                return state
        return CoordState.STARTING

    @property
    def generation(self) -> int:
        return max(c.generation for c in self.flights)

    async def start(self) -> None:
        check_shm_capacity(len(self.flights), self.flights[0].shm_size)
        for coord in self.flights:
            await coord.start()

    async def aclose(self) -> None:
        for coord in self.flights:
            await coord.aclose()


def num_flights() -> int:
    """How many coordinator processes to run.

    Read from the environment rather than a constant because the transport from
    `donutServer.py --num-flights` is the environment: uvicorn imports this module
    in-process, so a CLI flag can only reach here as an env var.
    """
    try:
        n = int(os.environ.get("DONUT_SERVER_NUM_FLIGHTS", "2"))
    except ValueError:
        n = 2
    return max(1, n)


# ------------------------------------------------------------------ plot flights

# The file stems coordinator.PLOT_FIGURES writes, which are also the only names
# /result/{job_id}/plot/{name} will serve. A whitelist, not a sanitizer: `name`
# arrives from a URL, and membership here is the single check that keeps it from
# reaching the filesystem as a path.
PLOT_NAMES = ("donut", "wavefront", "selection")
PLOT_SIZES = ("full", "thumb")

# How long a drain loop waits after finding its flight unusable, before taking
# the same job again. The flight's own bring-up has a backoff; this is the one
# that keeps _drain from spinning, because _ensure_usable refuses a RESTARTING or
# DEGRADED flight instantly rather than waiting. Sized against a ~30 s cadence:
# long enough that a terminally degraded flight costs nothing, short enough that
# a restart (~19 s) is picked up on the next tick or two.
PLOT_RETRY_S = 10.0

# How many times one job may be put back because its flight had no usable child,
# before it is failed instead. Bounded so a job cannot be retried forever: a
# DEGRADED flight is recoverable only via /admin/restart?pool=plot, and until
# someone does that the honest report is an error the page can show, not a
# `queued` the page polls indefinitely. At PLOT_RETRY_S this gives a restart
# (~19 s) several chances before giving up.
PLOT_MAX_RETRIES = 6

# How many failure messages PlotQueue remembers. Only read by the manifest, one
# job at a time, so this is a memory bound and nothing else.
PLOT_ERROR_CAP = 256

# coordinator.PLOT_TABLES' values. Duplicated rather than imported for the usual
# reason -- importing coordinator pulls the LSST stack into this process -- and
# needed here only so the backfill scan can tell a job's result table from the two
# selection tables sitting beside it under the same prefix.
_SELECTION_KINDS = ("detector_images", "selection_overlays")

# What this service accepts as a job id in a path segment. Every id it mints is a
# uuid4, but these arrive from a URL, and one of them is about to be joined onto
# a filesystem path -- so the shape is checked rather than assumed. Hyphens and
# hex only: no dots, no separators, so neither ".." nor "/" can survive.
_JOB_ID_OK = re.compile(r"[0-9a-zA-Z_-]{1,64}")


def num_plot_flights() -> int:
    """How many plot processes to run. Same environment transport as num_flights.

    Defaults to 1: drawing measured ~17 s against a ~30 s per-flight cadence, so
    one lane keeps up with a single Zernike flight and falls behind two. Raising
    this trades cores for backlog -- see PlotQueue on why falling behind is
    tolerated rather than prevented.
    """
    try:
        n = int(os.environ.get("DONUT_SERVER_NUM_PLOT_FLIGHTS", "1"))
    except ValueError:
        n = 1
    return max(0, n)


class PlotFlight(Coord):
    """A `Coord` that runs `plot_flight_main` and owns no shared block.

    Subclassed rather than reimplemented: everything this needs from Coord --
    spawn, the hello handshake, bounded restart with backoff, the terminal
    DEGRADED state, killpg reaping, and the /health projection -- is already
    there and already tested. The two differences are that there is no 513 MB
    block to create (raw pixels never come here; the tables are on disk) and
    nothing to re-prime, since a plot flight holds no config.
    """

    def _resolve_target(self):
        if self._target is not None:
            return self._target
        from lsst.ts.donut_server import coordinator

        self._target = coordinator.plot_flight_main
        return self._target

    async def start(self) -> None:
        # Coord.start() creates the shared block first; this deliberately does
        # not, leaving _shm None -- which is what makes _spawn_child pass the
        # connection alone, and makes aclose()'s unlink a no-op.
        self._state = CoordState.STARTING
        self._startup_task = asyncio.create_task(self._bring_up())

    async def _reprime(self) -> None:
        """Nothing to replay: a plot flight is stateless between jobs."""
        return


class PlotQueue:
    """Job ids waiting to be drawn, newest first, and the flights that drain them.

    **LIFO, and uncapped.** Newest-first because a backlog's most recent entry is
    the one an operator is most likely looking at, so that is what should be drawn
    when a flight frees up. Uncapped because observing has gaps in which a backlog
    drains, and dropping work would mean choosing, at enqueue time, which job will
    never be viewable.

    The consequence is accepted rather than prevented: with one flight at ~17 s a
    job against a ~15 s arrival rate, continuous observing grows the queue and its
    *tail* -- the oldest entries -- may never be reached. `depth` is on /health so
    that saturation is visible rather than silent, and `--num-plot-flights` is the
    dial. The tail costs nothing but a string: the 38 MB a job needs is already on
    disk, written by the coordinator for its own reasons.

    Not persisted. The queue dies with the front-end exactly as JOBS does;
    POST /admin/plots/backfill rebuilds it from what is on disk.
    """

    def __init__(self, flights: list[PlotFlight]) -> None:
        self.flights = flights
        # A list used as a stack, plus a set for O(1) dedupe. Both are needed: a
        # job already queued must not be enqueued twice by a page poll, and a job
        # already *drawing* must not be enqueued behind itself either.
        self._stack: list[str] = []
        self._queued: set[str] = set()
        self._drawing: dict[int, str] = {}
        self._wake = asyncio.Event()
        self._drains: list[asyncio.Task] = []
        # Terminal outcomes, for the manifest: a job whose plotting failed should
        # say so rather than reading as "never asked for".
        #
        # Bounded, unlike JOBS, which is paged instead. A front-end whose plot
        # flight is wedged or whose disk filled would otherwise accumulate one
        # entry per failure for the life of the process -- small individually,
        # unbounded in aggregate, and the oldest are the least interesting.
        # OrderedDict so the eviction is FIFO; see _record_error.
        self._errors: OrderedDict[str, str] = OrderedDict()
        # Consecutive "no usable child" retries per job, cleared on any terminal
        # outcome. Never larger than the number of jobs in flight across the
        # pool, so it needs no bound of its own.
        self._retries: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._stack)

    @property
    def drawing(self) -> dict[int, str]:
        return dict(self._drawing)

    def state_of(self, job_id: str) -> Optional[str]:
        """Where `job_id` is in this queue, or None if it is not in it at all."""
        if job_id in self._drawing.values():
            return "drawing"
        if job_id in self._queued:
            return "queued"
        return None

    def error_of(self, job_id: str) -> Optional[str]:
        return self._errors.get(job_id)

    def _record_error(self, job_id: str, message: str) -> None:
        """Remember why `job_id` could not be drawn, evicting the oldest if full."""
        self._errors.pop(job_id, None)  # re-insert at the end, so it is newest
        self._errors[job_id] = message
        while len(self._errors) > PLOT_ERROR_CAP:
            self._errors.popitem(last=False)

    def enqueue(self, job_id: str) -> bool:
        """Push `job_id` to the front of the line. False if it was not queued.

        A pool with no flights refuses rather than accumulating: nothing would
        ever drain it, and a queued-forever job makes the results page poll for a
        plot that is never coming. `--num-plot-flights 0` is a supported
        configuration, so this is the ordinary path, not an error -- the routes
        turn it into a 409 and the manifest reports `absent`.
        """
        if not self.flights:
            return False
        if job_id in self._queued or job_id in self._drawing.values():
            return False
        self._queued.add(job_id)
        self._stack.append(job_id)
        # A retry clears a previous failure, so the manifest stops reporting an
        # error the moment the job is genuinely pending again.
        self._errors.pop(job_id, None)
        self._wake.set()
        return True

    def _take(self) -> Optional[str]:
        """The newest queued job, or None. LIFO: pop the end."""
        while self._stack:
            job_id = self._stack.pop()
            if job_id in self._queued:
                self._queued.discard(job_id)
                return job_id
        return None

    async def start(self) -> None:
        for flight in self.flights:
            await flight.start()
        self._drains = [
            asyncio.create_task(self._drain(flight)) for flight in self.flights
        ]

    async def _drain(self, flight: PlotFlight) -> None:
        """One flight's whole working life: take the newest job, draw it, repeat.

        Every failure is swallowed into `_errors`. A plot that cannot be drawn
        must not end this loop, or one bad job would cost the flight for the rest
        of the process's life -- and nobody is blocked on a plot.
        """
        while True:
            job_id = self._take()
            if job_id is None:
                # clear() then re-check, in that order, and this is what makes a
                # shared Event safe for N flights. An Event wakes every waiter,
                # so one flight can clear a set() that another flight's work
                # caused -- the re-check is what stops that becoming a lost
                # wakeup. The invariant: a flight only sleeps having seen an
                # empty stack *after* clearing, so any enqueue before the clear
                # left an item the re-check finds, and any enqueue after it sets
                # the event again.
                self._wake.clear()
                if not self._stack:
                    await self._wake.wait()
                continue

            self._drawing[flight.index] = job_id
            try:
                resp = await flight.send_command({"cmd": "plot", "job_id": job_id})
                self._retries.pop(job_id, None)
                if not resp.get("ok"):
                    self._record_error(job_id, resp.get("error", "plotting failed"))
                else:
                    # One exposure drawn. Counted on the same field a Zernike
                    # flight uses for a completed push, so the dashboard reads
                    # "exposures" the same way in both sections.
                    flight.note_exposure()
            except asyncio.CancelledError:
                # Shutdown, so this is bookkeeping for the benefit of whatever
                # inspects the queue between the cancel and the exit -- the
                # process is going down and nothing reads it afterwards. Put back
                # rather than dropped so `depth` never under-reports work that
                # was taken but not done.
                self._queued.add(job_id)
                self._stack.append(job_id)
                raise
            except (CoordinatorLost, CoordinatorUnavailable) as exc:
                # The flight has no usable child: it died mid-draw, is restarting,
                # or has given up entirely (DEGRADED). Crucially _ensure_usable
                # raises *immediately* for the latter two, so without a backoff
                # this loop would spin at full speed -- taking the whole queue,
                # failing each job in microseconds, and burning a core for as long
                # as the flight stayed down.
                #
                # Retried a bounded number of times, then failed. Nothing about
                # the job went wrong, so a retry is right -- but DEGRADED is
                # terminal until /admin/restart?pool=plot, and an unbounded retry
                # would leave the job reporting `queued` for as long as nobody
                # noticed. An error the page can show is the honest report.
                tries = self._retries.get(job_id, 0) + 1
                if tries >= PLOT_MAX_RETRIES:
                    self._retries.pop(job_id, None)
                    self._record_error(
                        job_id, f"gave up after {tries} attempts: {type(exc).__name__}: {exc}"
                    )
                else:
                    self._retries[job_id] = tries
                    self._queued.add(job_id)
                    self._stack.append(job_id)
                    self._errors.pop(job_id, None)
                await asyncio.sleep(PLOT_RETRY_S)
            except Exception as exc:
                # A real failure to draw this job -- a corrupt table, a figure that
                # raised. Recorded against the job, which keeps it until retried,
                # and the loop moves on to the next one.
                self._retries.pop(job_id, None)
                self._record_error(job_id, f"{type(exc).__name__}: {exc}")
            finally:
                self._drawing.pop(flight.index, None)

    async def aclose(self) -> None:
        for task in self._drains:
            task.cancel()
        for task in self._drains:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._drains = []
        for flight in self.flights:
            await flight.aclose()


pool = FlightPool(num_flights())
plot_pool = PlotQueue(
    [PlotFlight(index=i) for i in range(num_plot_flights())]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.start()
    await plot_pool.start()
    yield
    await plot_pool.aclose()
    await pool.aclose()


app = FastAPI(lifespan=lifespan)

# Retry-After, in seconds, per failure reason.
_RETRY_AFTER = {
    "coordinator_lost": 30,
    "coordinator_restarting": 15,
    "coordinator_degraded": 3600,
}


async def _coordinator_error(request: Request, exc: Exception) -> JSONResponse:
    """503 for every coordinator-liveness failure, with the distinction in `reason`.

    503 rather than 502 even for the death case: every retry library treats
    503 + Retry-After as "come back", while 502 is often treated as
    semi-permanent, and what the producer actually needs is in `reason`.
    """
    # Coord tags what it raises with its own index, so the body describes the lane
    # that actually failed rather than the pool aggregate -- which, with N flights,
    # will often read READY while this request's flight is not.
    index = getattr(exc, "flight", None)
    source: Any = pool.flights[index] if index is not None else pool

    if isinstance(exc, CoordinatorLost):
        reason = "coordinator_lost"
    elif source.state is CoordState.DEGRADED:
        reason = "coordinator_degraded"
    else:
        reason = "coordinator_restarting"
    retry_after = _RETRY_AFTER[reason]
    return JSONResponse(
        status_code=503,
        content={
            "reason": reason,
            "error": str(exc),
            "state": source.state.value,
            "generation": source.generation,
            "flight": index,
        },
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


app.add_exception_handler(CoordinatorLost, _coordinator_error)
app.add_exception_handler(CoordinatorUnavailable, _coordinator_error)


def _is_loopback(request: Request) -> bool:
    """Whether the peer is on this machine.

    Safe to trust rather than spoofable. uvicorn rewrites scope["client"] from
    X-Forwarded-For only when the immediate peer is already within
    forwarded_allow_ips, which defaults to 127.0.0.1 -- so a remote caller sending
    that header keeps its own address and stays remote (verified against uvicorn
    1.6.0). A loopback reverse proxy forwarding a real remote client correctly
    yields *that* client's address, which then needs a token.

    ipaddress rather than a compare against "127.0.0.1": the loopback block is all
    of 127/8, and ::1 and ::ffff:127.0.0.1 are the same machine too. A peer that is
    not an address at all (a unix socket has none) raises, which is not local.
    """
    host = request.client.host if request.client else None
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_auth(request: Request, authorization: str = Header(None)) -> None:
    """Bearer token, except from this machine.

    A caller on loopback is already past the only boundary that matters: the
    dashboard is reached either on the server box or through an ssh tunnel, and a
    tunnel terminates here and reconnects to 127.0.0.1, so both are loopback. What
    this deliberately does not defend against is another user with shell access on
    this host -- accepted, because such a user can read DONUT_SERVER_TOKEN out of
    the environment anyway.

    Remote callers are unaffected: the producers pushing pixels to /prepare and
    /push still need the token.
    """
    if _is_loopback(request):
        return
    token = os.environ.get("DONUT_SERVER_TOKEN")
    if not token:
        raise HTTPException(500, "server missing DONUT_SERVER_TOKEN")
    if authorization != f"Bearer {token}":
        raise HTTPException(401, "unauthorized")


def require_local(request: Request) -> None:
    """Loopback or nothing -- no token fallback, unlike check_auth.

    For the two HTML pages. They hold no data themselves, but serving them to a
    remote browser would render a dashboard whose every fetch then fails, which
    reads as a broken page rather than a closed door.
    """
    if not _is_loopback(request):
        raise HTTPException(403, "the dashboard is available from localhost only")


def _required_float(body: dict, key: str) -> float:
    """Reject a missing or non-numeric field with a 400 rather than letting it
    reach the coordinator, where it would surface as a 500.

    `json` parses the bare `NaN` token, so finiteness has to be checked here too.
    """
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, f"{key} is required and must be a number")
    if not math.isfinite(value):
        raise HTTPException(400, f"{key} must be finite, got {value!r}")
    return float(value)


def _required_str(body: dict, key: str) -> str:
    """Reject a missing, non-string or blank field with a 400 rather than letting
    it reach the coordinator.

    Shape only -- whether calibs actually exist for the value is the
    coordinator's to answer, and it says so with the path it looked for.
    """
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(400, f"{key} is required and must be a non-empty string")
    return value


def _parse_overrides(body: dict) -> list:
    """Validate and normalize `config_overrides`. Returns [] when absent.

    Shape only -- never field names or values. Checking whether `maxFitScatter` is a
    real field would mean importing DonutBlitzCornerConfig, and with it the whole
    LSST stack, into this process (see _stamp_path). The coordinator owns that check
    and reports it as a 400 via its `kind` tag.

    Absent, null and [] are deliberately identical: a prepare states the config in
    full, so a bare /prepare is how an operator resets a bad override set.
    """
    spec = body.get("config_overrides")
    if spec is None:
        return []
    if not isinstance(spec, list):
        raise HTTPException(400, "config_overrides must be a list")
    if len(spec) > MAX_OVERRIDES:
        raise HTTPException(400, f"too many config_overrides (max {MAX_OVERRIDES})")

    total = 0
    out = []
    for i, entry in enumerate(spec):
        if not isinstance(entry, dict):
            raise HTTPException(400, f"config_overrides[{i}] must be an object")
        kind = entry.get("kind")
        if kind not in OVERRIDE_KINDS:
            # Never ignored: a typo'd kind that silently ran with defaults would be
            # the worst outcome here, since the job still succeeds.
            raise HTTPException(
                400,
                f"config_overrides[{i}].kind must be one of {list(OVERRIDE_KINDS)}, "
                f"got {kind!r}",
            )

        if kind == "value":
            field, value = entry.get("field"), entry.get("value")
            if not isinstance(field, str) or not field.strip():
                raise HTTPException(400, f"config_overrides[{i}].field must be a non-empty string")
            # Must stay a string end to end: the coordinator relies on that to get
            # pipetask's command-line parsing rather than YAML's. See apply_overrides.
            if not isinstance(value, str):
                raise HTTPException(
                    400,
                    f"config_overrides[{i}].value must be a string (got "
                    f"{type(value).__name__}); send \"2.0\", not 2.0",
                )
            total += len(field) + len(value)
            out.append({"kind": "value", "field": field, "value": value})
        else:
            text, name = entry.get("text"), entry.get("name")
            if not isinstance(text, str):
                raise HTTPException(400, f"config_overrides[{i}].text must be a string")
            if name is not None and not isinstance(name, str):
                raise HTTPException(400, f"config_overrides[{i}].name must be a string")
            size = len(text.encode())
            if size > MAX_OVERRIDE_TEXT_BYTES:
                raise HTTPException(
                    400,
                    f"config_overrides[{i}].text is {size} bytes, over the "
                    f"{MAX_OVERRIDE_TEXT_BYTES} byte limit",
                )
            total += size
            out.append({"kind": "python", "name": name or "<override>", "text": text})

        if total > MAX_OVERRIDE_TOTAL_BYTES:
            raise HTTPException(
                400, f"config_overrides total exceeds {MAX_OVERRIDE_TOTAL_BYTES} bytes"
            )
    return out


def _digest_overrides(spec: Optional[list]) -> list:
    """Echo-safe projection of an override list: no `-C` bodies.

    /health carries this and is polled once a second, so a 50 KB override file echoed
    verbatim would be 50 KB/s forever. `value` entries are small and are what an
    operator reads off a dashboard, so they survive whole; a `python` entry becomes
    its identity plus a fingerprint, and its body is recoverable from GET /config,
    whose full config dump says strictly more anyway.
    """
    digest = []
    for entry in spec or ():
        if entry.get("kind") == "value":
            digest.append(dict(entry))
        else:
            text = entry.get("text") or ""
            digest.append({
                "kind": "python",
                "name": entry.get("name") or "<override>",
                "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
                "bytes": len(text.encode()),
                "sha256": hashlib.sha256(text.encode()).hexdigest()[:12],
            })
    return digest


def _redact_primed_args(args: Optional[dict]) -> Optional[dict]:
    """primed_args for display. Everything but the override bodies is passed through.

    A projection rather than a redaction in place: _primed_args itself has to keep
    the full `-C` text, because that dict is what gets replayed to a fresh
    coordinator after a restart.
    """
    if args is None:
        return None
    shown = dict(args)
    if "config_overrides" in shown:
        shown["config_overrides"] = _digest_overrides(shown["config_overrides"])
    return shown


@app.post("/prepare", dependencies=[Depends(check_auth)])
async def prepare(body: dict):
    # The boresight is required: it is what lets the coordinator pre-load
    # reference catalog shards for the pointing before any pixels exist,
    # keeping ~450 ms of shard resharding off the push critical path.
    boresight_ra = _required_float(body, "boresight_ra")
    boresight_dec = _required_float(body, "boresight_dec")
    overrides = _parse_overrides(body)

    command = {
        "cmd": "prepare",
        "physical_filter": _required_str(body, "physical_filter"),
        "boresight_ra": boresight_ra,
        "boresight_dec": boresight_dec,
        # When the exposure was taken, as ISO-8601 TAI. Required, and validated
        # for shape only: parsing it needs astropy, which this process
        # deliberately never imports (see _stamp_path), so the coordinator does
        # that half and reports a bad value with kind="calib_time".
        #
        # Required rather than optional because it selects calibs under the
        # butler backend, and a prepare that omitted it would resolve against no
        # time at all -- which in daf_butler means excluding every CALIBRATION
        # collection, i.e. silently wrong calibs rather than an error.
        "calib_time": _required_str(body, "calib_time"),
        # Sent even when empty, so _primed_args is never ambiguous about whether
        # a replayed prepare had overrides.
        "config_overrides": overrides,
    }
    job_id = str(uuid.uuid4())
    coord = pool.select(command)
    # Routed before the command is sent, not after: a prepare that kills its child
    # must already be attributable to a lane, or _fail_jobs would skip it.
    pool.route(job_id, coord)
    async with pool.queued(coord):
        resp = await coord.send_command(command)
    if not resp.get("ok"):
        JOBS[job_id] = JobRecord(job_id=job_id, state=JobState.ERROR, error=resp.get("error"))
        # 400 for a bad override, because it will fail identically forever; 500 would
        # tell a producer's retry logic "server broken, come back". The same
        # transient-vs-permanent distinction _RETRY_AFTER draws for 503s.
        status = 400 if resp.get("kind") in ("config_override", "calib_time") else 500
        raise HTTPException(status, resp.get("error", "prepare failed"))

    JOBS[job_id] = JobRecord(
        job_id=job_id,
        state=JobState.PREPARED,
        prepare_timings=resp.get("timings"),
        prepared_key=resp.get("prepared_key"),
    )
    # `generation` lets the producer's own logs attribute a latency spike to a
    # coordinator restart. The override digest is echoed so a --loop producer's own
    # log records the config each cycle actually ran with.
    return {
        "job_id": job_id,
        "state": JobState.PREPARED,
        "timings": resp.get("timings"),
        "generation": coord.generation,
        # Which lane served this, so a producer's own logs can attribute latency to
        # a flight rather than to the service as a whole.
        "flight": coord.index,
        "config_overrides": _digest_overrides(overrides),
    }


@app.post("/push/{job_id}", dependencies=[Depends(check_auth)])
async def push(job_id: str, request: Request, num_workers: int | None = None):
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")
    if record.state != JobState.PREPARED:
        raise HTTPException(409, f"job in state {record.state}, expected PREPARED")
    if num_workers is not None and num_workers < 1:
        raise HTTPException(400, "num_workers must be >= 1")

    # The flight that prepared this job: its child is the one holding the config,
    # and its shared block is the one the pixels have to land in.
    coord = pool.for_push(job_id)
    async with pool.queued(coord), coord.push_lock:
        pool.note_used(coord)
        view = memoryview(coord.shm.buf)
        try:
            return await _receive_and_dispatch(
                coord, job_id, record, request, view, num_workers
            )
        finally:
            # Must be released before Coord.stop() can close the block.
            view.release()


async def _receive_and_dispatch(
    coord: Coord, job_id, record, request: Request, view: memoryview,
    num_workers: int | None = None,
) -> dict:
    """Stream the body into `view`, then hand the coordinator just the layout."""
    body_iter = request.stream()
    received = 0

    def absorb(chunk: bytes) -> None:
        nonlocal received
        end = received + len(chunk)
        if end > len(view):
            raise HTTPException(413, "payload too large")
        view[received:end] = chunk
        received = end

    t_receive = time.perf_counter()
    async for chunk in body_iter:
        absorb(chunk)
        if received >= protocol.HEADER_SIZE:
            break

    if received < protocol.HEADER_SIZE:
        raise HTTPException(400, "body too short for header")
    try:
        header = protocol.parse_header(view[: protocol.HEADER_SIZE])
    except protocol.ProtocolError as exc:
        raise HTTPException(400, str(exc))
    if header.payload_length > MAX_PUSH_BYTES:
        raise HTTPException(413, "payload too large")

    record.state = JobState.RECEIVING
    async for chunk in body_iter:
        absorb(chunk)
    receive_s = time.perf_counter() - t_receive

    t_layout = time.perf_counter()
    try:
        layout = protocol.parse_layout(view[:received])
    except protocol.ProtocolError as exc:
        record.state = JobState.ERROR
        record.error = str(exc)
        raise HTTPException(400, str(exc))
    layout_s = time.perf_counter() - t_layout

    record.state = JobState.COMPUTING
    # Only the layout crosses the Pipe; the pixels stay in shared memory.
    try:
        resp = await coord.send_command(
            {
                "cmd": "push",
                "job_id": job_id,
                "layout": layout,
                "num_workers": num_workers,
                "prepared_key": record.prepared_key,
            }
        )
    except (CoordinatorLost, CoordinatorUnavailable) as exc:
        # Record the failure before the 503 goes out, so /status and /result agree
        # with what the client just saw rather than stranding this job in COMPUTING.
        record.state = JobState.ERROR
        record.error = str(exc)
        raise
    if not resp.get("ok"):
        record.state = JobState.ERROR
        record.error = resp.get("error")
        raise HTTPException(500, resp.get("error", "push failed"))

    result = resp.get("result") or {}
    timings = {
        "blob_bytes": received,
        "receive_s": receive_s,
        "layout_s": layout_s,
        **result.get("timings", {}),
    }
    record.state = JobState.DONE
    record.table_parquet = result.pop("table_parquet", None)
    record.result = result
    record.push_timings = timings
    # Here rather than at /prepare: one exposure can be prepared and never
    # pushed, or prepared on a lane that an ERROR then takes it off, and neither
    # is work this flight did.
    coord.note_exposure()
    # Enqueued, not awaited: the tables this needs are written by the
    # coordinator's deferred step, which runs *after* the reply we just received,
    # so the plot flight will find them by the time it gets there -- and if it
    # races ahead, render_plots raises FileNotFoundError and the job can be
    # retried from the page. Pushing to a LIFO means this newest job is next.
    plot_pool.enqueue(job_id)
    return {
        "job_id": job_id,
        "state": record.state,
        "timings": timings,
        "generation": coord.generation,
        "flight": coord.index,
    }


@app.get("/status/{job_id}", dependencies=[Depends(check_auth)])
async def status(job_id: str):
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")
    return {"job_id": job_id, "state": record.state}


@app.get("/result/{job_id}", dependencies=[Depends(check_auth)])
async def result(job_id: str, wait: float = 0.0):
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")

    deadline = time.monotonic() + wait
    while record.state not in (JobState.DONE, JobState.ERROR):
        if time.monotonic() >= deadline:
            return {"ready": False, "state": record.state}
        await asyncio.sleep(0.1)

    if record.state == JobState.ERROR:
        return {"ready": True, "state": record.state, "error": record.error}
    return {
        "ready": True,
        "state": record.state,
        "result": record.result,
        "timings": record.push_timings,
        "table_url": f"/result/{job_id}/table",
        # Advertised unconditionally rather than gated on the file existing: this
        # reply is what a client long-polls for, and it can be sent while the
        # coordinator is still writing the images. The URL is the durable fact;
        # whether the bytes have landed yet is what GETting it tells you.
        "images_url": f"/result/{job_id}/images",
    }


@app.get("/result/{job_id}/table", dependencies=[Depends(check_auth)])
async def result_table(job_id: str):
    """The wavefront results as parquet, byte-identical to what a Butler would
    write for a `donutBlitzCornerResults` dataset.

    Kept separate from `/result` so that endpoint stays JSON and long-pollable.
    """
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")
    if record.state == JobState.ERROR:
        raise HTTPException(409, record.error or "job failed")
    if record.table_parquet is None:
        raise HTTPException(409, f"no table for job in state {record.state}")
    return Response(
        content=record.table_parquet,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.parquet"'},
    )


@app.get("/result/{job_id}/images", dependencies=[Depends(check_auth)])
async def result_images(job_id: str):
    """The same table *with* the ~17 MB of per-donut stamps, read off disk.

    The coordinator writes this after it has already answered the push, so a 409
    here is the ordinary state for a few tens of milliseconds after a job reports
    DONE -- not an error, just early. Retry rather than treat it as missing.

    Served as an opaque file: this process no more knows what a stamp is than it
    knows what a Zernike is, which is what keeps astropy and the LSST stack out
    of it. The write is a tmp-plus-os.replace, so a visible path is a complete
    file and there is no partial-read window to guard against.
    """
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")
    if record.state == JobState.ERROR:
        raise HTTPException(409, record.error or "job failed")
    path = _stamp_path(job_id)
    if path is None or not os.path.exists(path):
        raise HTTPException(409, f"no images yet for job in state {record.state}")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{job_id}.parquet",
    )


@app.get("/result/{job_id}/plots", dependencies=[Depends(check_auth)])
async def result_plots(response: Response, job_id: str):
    """Which diagnostic figures exist for this job, and why not if they do not.

    The manifest the results page polls. It reports *state* rather than just
    presence because every absence here has a different remedy: `queued` and
    `drawing` resolve themselves, `error` wants the log, and `absent` wants
    someone to ask for the plots (POST /admin/plots/{job_id}).

    Not gated on the job being DONE. A plot can legitimately outlive the front-end
    that computed it -- the PNGs are on disk and JOBS is not -- so a job this
    process has never heard of still gets its files reported rather than a 404.
    """
    # Validated like every other job-id route here: this one joins `job_id` onto
    # the stamp dir too, and while a single path segment holds no separator, an
    # unchecked one still probes for the existence and size of files outside the
    # plots tree.
    if not _JOB_ID_OK.fullmatch(job_id):
        raise HTTPException(400, "malformed job id")

    response.headers.update(_NO_STORE)
    record = JOBS.get(job_id)

    figures = {}
    for name in PLOT_NAMES:
        sizes = {}
        for size in PLOT_SIZES:
            path = _plot_path(job_id, name, size)
            if path is not None and os.path.exists(path):
                stat = os.stat(path)
                sizes[size] = {
                    "bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "url": f"/result/{job_id}/plot/{name}?size={size}",
                }
        if sizes:
            figures[name] = sizes

    queue_state = plot_pool.state_of(job_id)
    error = plot_pool.error_of(job_id)
    # Order matters, and `error` comes before `ready` deliberately. A *failed
    # redraw* of a job that already has figures would otherwise report "ready"
    # and the operator would read six stale PNGs as current -- the figures on
    # disk are from the previous successful draw, and the thing they just asked
    # for did not happen. Pending still outranks both: a redraw in flight should
    # keep the page polling rather than settle on either verdict.
    if queue_state is not None:
        state = queue_state
    elif error is not None:
        state = "error"
    elif figures:
        state = "ready"
    else:
        state = "absent"

    return {
        "job_id": job_id,
        "state": state,
        "error": error,
        "figures": figures,
        # So the page can say "3rd in line" rather than only "queued", and can
        # tell a disabled plot pool from a saturated one.
        "queue_depth": len(plot_pool),
        "n_plot_flights": len(plot_pool.flights),
        "job_state": None if record is None else record.state,
    }


@app.get("/result/{job_id}/plot/{name}", dependencies=[Depends(check_auth)])
async def result_plot(job_id: str, name: str, size: str = "full"):
    """One rendered PNG.

    `name` and `size` are checked against closed whitelists *before* either
    reaches `_plot_path`. That is the whole defence: both arrive from a URL, and
    `job_id` too, so without it a crafted path segment would read any file the
    service user can. 400 rather than 404 for an unknown name -- it is a malformed
    request, not a missing figure.
    """
    if name not in PLOT_NAMES:
        raise HTTPException(400, f"unknown plot {name!r}; expected one of {list(PLOT_NAMES)}")
    if size not in PLOT_SIZES:
        raise HTTPException(400, f"unknown size {size!r}; expected one of {list(PLOT_SIZES)}")
    # A job id is a uuid everywhere this service mints one, but it arrives here as
    # a URL segment like the others, so it gets the same treatment rather than
    # being trusted for its shape.
    if not _JOB_ID_OK.fullmatch(job_id):
        raise HTTPException(400, "malformed job id")

    path = _plot_path(job_id, name, size)
    if path is None or not os.path.exists(path):
        # 409 for the same reason /result/{job_id}/images answers 409: for the ~17 s
        # after a job is DONE this is the ordinary state, not a failure. The
        # manifest above says which it is.
        raise HTTPException(409, f"no {name} plot for job {job_id} yet")
    return FileResponse(path, media_type="image/png")


@app.post("/admin/plots/{job_id}", dependencies=[Depends(check_auth)])
async def admin_plot_job(response: Response, job_id: str):
    """Draw (or redraw) one job's plots.

    How a job gets plotted when nothing enqueued it automatically: the queue is
    in-memory, so a front-end restart drops it, and `--num-plot-flights 0` never
    had one. Also the retry path for a job whose plotting failed.

    202, not 200: this returns as soon as the id is on the stack, which is the
    only thing it can honestly report. `queued: false` means the job was already
    pending, not that the request failed.
    """
    if not _JOB_ID_OK.fullmatch(job_id):
        raise HTTPException(400, "malformed job id")
    if not plot_pool.flights:
        raise HTTPException(
            409,
            "no plot flights are running; start the server with "
            "--num-plot-flights 1 or more",
        )
    queued = plot_pool.enqueue(job_id)
    response.status_code = 202
    response.headers.update(_NO_STORE)
    return {
        "job_id": job_id,
        "queued": queued,
        "state": plot_pool.state_of(job_id),
        "queue_depth": len(plot_pool),
    }


def _scan_for_missing_plots(limit: int) -> tuple[int, list[str]]:
    """How many jobs are on disk, and which of them have no plots.

    Shared by the two routes below, which differ only in what they do with the
    answer: the GET reports it, the POST enqueues it. Keeping one scan means the
    dashboard's "12 missing" and the number the button acts on cannot disagree.

    Returns the candidate count -- every job in the directory, not just the
    `limit` examined -- alongside the missing ids newest-first.
    """
    directory = os.environ.get("DONUT_SERVER_STAMP_DIR")
    if not directory:
        raise HTTPException(409, "DONUT_SERVER_STAMP_DIR is not set; nothing to scan")

    # Sync file I/O on the event loop, unlike the log and table routes. Justified
    # by scale rather than by nature: this is one listdir plus a stat per entry on
    # a directory with hundreds of files, and the alternative -- a threadpool hop
    # -- would need the routes to be sync `def`, which would then hold a worker
    # for the whole scan. The GET is polled rather than operator-triggered, which
    # is why the dashboard runs it on a 5 s loop rather than the 1 Hz the rest of
    # the page uses, and refreshes it on the events that can change the answer.
    candidates = []
    try:
        for entry in os.scandir(directory):
            if not entry.is_file() or not entry.name.endswith(".parquet"):
                continue
            job_id = entry.name[: -len(".parquet")]
            # The selection tables live in the same directory and share the
            # prefix, so they would otherwise read as two more jobs.
            if any(job_id.endswith(f".{kind}") for kind in _SELECTION_KINDS):
                continue
            candidates.append((entry.stat().st_mtime, job_id))
    except OSError as exc:
        raise HTTPException(409, f"cannot scan {directory}: {exc}") from exc

    candidates.sort(reverse=True)
    missing = [
        job_id
        for _, job_id in candidates[:limit]
        # "No plots" means no full-res donut plot: it is the one figure every job
        # with any rows produces, so its absence is the cheapest reliable test.
        if not os.path.exists(_plot_path(job_id, "donut", "full") or "")
    ]
    return len(candidates), missing


@app.get("/admin/plots", dependencies=[Depends(check_auth)])
async def admin_plot_survey(response: Response, limit: int = 200):
    """What the POST below would do, without doing it.

    Exists so the dashboard's backfill note is live rather than frozen at
    whatever the last click returned: `missing` falls as the queue drains and
    rises as new jobs finish, and a number that only moves on click reads as
    current when it is not.

    Unlike the POST this does not require a plot flight. The scan is read-only,
    and "14 missing" is exactly what an operator running `--num-plot-flights 0`
    wants to see before deciding to restart with a flight.
    """
    limit = max(1, min(JOBS_PAGE_MAX, limit))
    scanned, missing = _scan_for_missing_plots(limit)

    response.headers.update(_NO_STORE)
    return {
        "scanned": scanned,
        "missing": len(missing),
        "queue_depth": len(plot_pool),
    }


@app.post("/admin/plots", dependencies=[Depends(check_auth)])
async def admin_plot_backfill(response: Response, limit: int = 200):
    """Enqueue every job that has a result table on disk but no plots.

    The bulk form of the per-job route above, for the case a front-end restart
    dropped a backlog: clicking one job at a time is not a workflow. Driven off
    the disk rather than off JOBS, so it also covers jobs computed by a previous
    process.

    Enqueued oldest-first so that the LIFO stack pops *newest*-first -- the push
    order has to be inverted for the drain order to come out useful. `limit`
    bounds how many are taken, newest by mtime, since a long-lived stamp dir can
    hold hundreds of jobs and the oldest are the least likely to be wanted.
    """
    if not plot_pool.flights:
        raise HTTPException(
            409,
            "no plot flights are running; start the server with "
            "--num-plot-flights 1 or more",
        )
    limit = max(1, min(JOBS_PAGE_MAX, limit))
    scanned, missing = _scan_for_missing_plots(limit)

    # Reversed, so the newest of them is pushed last and popped first.
    queued = [job_id for job_id in reversed(missing) if plot_pool.enqueue(job_id)]

    response.status_code = 202
    response.headers.update(_NO_STORE)
    return {
        "scanned": scanned,
        "missing": len(missing),
        "queued": len(queued),
        "queue_depth": len(plot_pool),
    }


def _config_is_stale(coord: Coord, snap: ConfigSnapshot) -> bool:
    """Whether the cached dump might not describe the coordinator running right now.

    Either the snapshot predates the live child (a restart whose re-prime did not
    succeed), or the coordinator is currently unprimed.
    """
    return snap.generation != coord.generation or coord.primed_args is None


@app.get("/config", dependencies=[Depends(check_auth)])
async def config_dump():
    """The blitz task's full config, as loadable Python, from the last prepare.

    This is the answer to "what did that job actually run with", and the counterpart
    to /prepare's `config_overrides`: the overrides say what was *asked for*, this
    says what the task ended up holding -- every field, including the ~28 defaults
    nobody overrode.

    Served from the front-end's cache, never by asking the coordinator: the command
    loop is serial, so a round-trip here would block behind an in-flight push, and a
    cache still answers when the coordinator is RESTARTING or DEGRADED -- which is
    when this question gets asked. Staleness is reported rather than hidden.

    With N flights there are N snapshots. The freshest non-stale one is served, and
    the flight it came from is named in a header, rather than merging them into a
    config that describes no real process.
    """
    candidates = [
        (c, c.config_snapshot) for c in pool.flights if c.config_snapshot is not None
    ]
    if not candidates:
        # 409, not 404 (the route exists) and not 503 (the coordinator may be
        # perfectly healthy, just unprepared). Same "real, but not yet" semantics as
        # /result/{job_id}/images.
        raise HTTPException(
            409, f"no config yet: coordinator is {pool.state.value} and has not prepared"
        )
    # Newest first, preferring a snapshot that still describes its live child.
    coord, snap = min(
        candidates, key=lambda pair: (_config_is_stale(*pair), -pair[1].at)
    )
    # Every header goes on this object: mutating an injected `response: Response` has
    # no effect once a handler returns a Response of its own.
    return Response(
        content=snap.dump,
        media_type="text/plain; charset=utf-8",
        headers={
            # A loadable config file, so a saved copy works with Config.load. That is
            # also why staleness rides in headers rather than a JSON envelope.
            "Content-Disposition": 'inline; filename="donutBlitzCornerConfig.py"',
            "X-Donut-Generation": str(snap.generation),
            "X-Donut-Config-Sha256": snap.sha256,
            "X-Donut-Flight": str(coord.index),
            "X-Donut-Stale": "1" if _config_is_stale(coord, snap) else "0",
            "Cache-Control": "no-store",
        },
    )


def _job_counts() -> dict:
    counts: dict[str, int] = {}
    for record in JOBS.values():
        counts[record.state.value] = counts.get(record.state.value, 0) + 1
    return counts


def _flight_health(coord: Coord, queue_depth: Optional[int] = None) -> dict:
    """One flight's /health projection. Serves both pools.

    `queue_depth` is passed rather than looked up because a plot flight's index
    names a lane in a different pool: `pool.queue_depth(coord)` keys on the index
    alone, so a plot flight would silently report the backlog of the *Zernike*
    flight that happens to share its number.
    """
    snap = coord.config_snapshot
    return {
        "index": coord.index,
        "status": coord.state.value,
        "ready": coord.state is CoordState.READY,
        "alive": coord.is_alive(),
        "primed": coord.primed_args is not None,
        # Projected, not raw: primed_args retains full -C bodies so it stays
        # replayable, and this endpoint is polled once a second.
        "primed_args": _redact_primed_args(coord.primed_args),
        # A ~200-byte fingerprint of the config, so the dashboard can show that
        # one is installed (and whether it is stale) without pulling 77 KB.
        "config": None if snap is None else {
            "generation": snap.generation,
            "stale": _config_is_stale(coord, snap),
            "bytes": len(snap.dump),
            "sha256": snap.sha256[:12],
            "n_overrides": len(snap.overrides),
            "at": snap.at,
        },
        "generation": coord.generation,
        "spawns": coord.spawns,
        # How much work this lane has done since the front-end came up, which is
        # the one number here that is about throughput rather than health. It
        # counts completed pushes for a Zernike flight and completed draws for a
        # plot flight -- one exposure either way.
        "exposures": coord.exposures,
        "restart_attempts": coord.restart_attempts,
        "pid": coord.child_pid,
        # The coordinator's own RSS, which on this service oscillates 4-6 GB as
        # calibs land and are dropped. Excludes the task's fork workers on
        # purpose; see rss_bytes.
        "rss_bytes": rss_bytes(coord.child_pid),
        "child_uptime_s": coord.child_uptime_s,
        # last_loss.exitcode is the highest-value field here: -11 is a segfault
        # (a fork-safety regression), -9 is SIGKILL, which on a service that
        # oscillates 4-6 GB RSS should be read as macOS jetsam, and 1 is a
        # Python exception with a traceback in the log.
        "last_loss": coord.last_loss,
        "degraded_reason": coord.degraded_reason,
        # Non-null means the service is misconfigured rather than unlucky, and
        # the process is on its way down; see Coord.fatal_reason.
        "fatal_reason": coord.fatal_reason,
        # alive without ready and with in_flight > 0, held for minutes, is the
        # wedged-but-alive signature that /admin/restart exists for.
        "in_flight": coord.in_flight,
        "busy": coord.busy,
        "queue_depth": queue_depth,
    }


@app.get("/health")
async def health(request: Request, response: Response, authorization: str = Header(None)):
    """The status *code* carries the verdict: 200 when any flight is READY, else 503.

    Nothing can supervise an endpoint that always returns 200. Auth is optional
    rather than absent -- a bare remote probe gets the verdict plus a minimal body,
    while a local caller or a valid token gets the full detail.

    "Any flight" rather than "all": a pool with one healthy lane still serves every
    request correctly, just at lower throughput, and a supervisor must not kill it
    for that. `status` says "partial" in exactly that case, so the degradation is
    still visible to anyone reading more than the status code.
    """
    n_ready = sum(1 for c in pool.flights if c.state is CoordState.READY)
    ready = n_ready > 0
    response.status_code = 200 if ready else 503
    response.headers["Cache-Control"] = "no-store"

    if ready and n_ready < len(pool):
        status = "partial"
    else:
        status = pool.state.value

    body: dict[str, Any] = {"status": status, "ready": ready}
    if not _is_loopback(request):
        token = os.environ.get("DONUT_SERVER_TOKEN")
        if not token or authorization != f"Bearer {token}":
            return body

    body.update(
        {
            "n_flights": len(pool),
            "n_ready": n_ready,
            # This process: the uvicorn front-end, which owns the job history and
            # every shared block, and whose pid is the one to send a signal to.
            # Its RSS is small and flat by design -- the parquet it retains is the
            # only thing that grows -- so a front-end climbing alongside the
            # flights is itself the finding.
            "front_end": {"pid": os.getpid(), "rss_bytes": rss_bytes(os.getpid())},
            "flights": [_flight_health(c, pool.queue_depth(c)) for c in pool.flights],
            "jobs": _job_counts(),
            # Reusing _flight_health rather than a parallel projection: a plot
            # flight *is* a Coord, so every field it reports means the same thing.
            # The config and prime fields are simply always null, since a plot
            # flight holds no config -- which is itself worth seeing.
            # The pool-wide depth on every row, not a per-flight split: one LIFO
            # stack feeds every plot flight, so there is no per-lane backlog.
            "plot_flights": [
                _flight_health(c, len(plot_pool)) for c in plot_pool.flights
            ],
            "plot_queue": {
                "depth": len(plot_pool),
                # By flight index, so the dashboard can label the row that is busy
                # rather than just counting.
                "drawing": {str(k): v for k, v in plot_pool.drawing.items()},
            },
        }
    )
    return body


@app.post("/admin/restart", dependencies=[Depends(check_auth)])
async def admin_restart(
    response: Response,
    flight: int | None = None,
    # `pool` in the query string; renamed in Python because the module-global
    # `pool` is the Zernike FlightPool and shadowing it here would be a trap.
    pool_name: str = Query("zernike", alias="pool"),
):
    """Recover a wedged-but-alive coordinator, and the only way out of DEGRADED.

    There is deliberately no automatic command timeout: a generous bound cannot be
    sized safely against a 6.1-8.9 s push, and a mis-sized one kills healthy work.
    An operator (or an external watchdog) who knows a job has been COMPUTING for
    minutes has strictly more information than a blind timeout.

    Bare, this restarts every flight, which is what it meant when there was only
    one. `?flight=N` targets a single lane -- the usual case now, since a wedge is
    a property of one child and restarting the healthy lanes alongside it would
    throw away their warm caches and any job they are computing.

    `?pool=plot` targets the plot flights instead, and it is not optional
    completeness: `Coord` treats DEGRADED as terminal *until* a restart, so
    without a way to name them here a plot flight that exhausted its attempts
    would stay down for the life of the process, with its drain loop retrying a
    job forever and the results page polling `queued` just as long.
    """
    if pool_name not in ("zernike", "plot"):
        raise HTTPException(400, f"pool must be 'zernike' or 'plot', got {pool_name!r}")
    flights = pool.flights if pool_name == "zernike" else plot_pool.flights
    if not flights:
        raise HTTPException(409, f"no {pool_name} flights are running")
    if flight is not None and not 0 <= flight < len(flights):
        raise HTTPException(
            400, f"flight must be in 0..{len(flights) - 1} for pool {pool_name!r}, got {flight}"
        )
    targets = flights if flight is None else [flights[flight]]
    reason = "operator requested restart via /admin/restart"
    response.status_code = 202
    return {
        "restarting": True,
        "pool": pool_name,
        "flights": [
            {
                "index": coord.index,
                "killed_pid": coord.request_restart(reason),
                "state": coord.state.value,
            }
            for coord in targets
        ],
    }


# ------------------------------------------------------------------- dashboard

_NO_STORE = {"Cache-Control": "no-store"}

_ACTIVE_STATES = (JobState.PREPARED, JobState.RECEIVING, JobState.COMPUTING)


@app.get("/dashboard", include_in_schema=False, dependencies=[Depends(require_local)])
async def dashboard():
    """The monitoring page. Localhost only, and carries no token of its own.

    Every byte the page displays arrives via the endpoints below, which exempt
    loopback for the same reason this route does. FileResponse rather than reading
    the file inline, so an edit to the page is picked up without restarting uvicorn
    -- which matters because restarting uvicorn here costs a fresh 15 s coordinator
    bring-up and the whole job history.
    """
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


@app.get("/results/{job_id}", include_in_schema=False, dependencies=[Depends(require_local)])
async def results_page(job_id: str):
    """The results viewer, opened in a second tab from the dashboard's job table.

    Same shape as /dashboard: a static shell, localhost only, that fetches its rows
    from the endpoint below. `job_id` is not used here -- the page reads it out of
    its own URL -- and deliberately never reaches the filesystem.
    """
    return FileResponse(RESULTS_HTML, media_type="text/html")


def _job_row(job_id: str, record: JobRecord, now: float) -> dict:
    """One row for /admin/jobs.

    `table_parquet` is deliberately never read: at ~200 KB a record, a page of 50
    polled once a second would be 10 MB/s of base64. Its size is already an int in
    the summary (coordinator.py:520).
    """
    result = record.result or {}
    summary = result.get("summary") or {}
    return {
        "job_id": job_id,
        # Which lane ran it. Null for a job whose route the front-end no longer
        # has, which is not an error -- see FlightPool.for_push.
        "flight": pool.flight_of(job_id),
        "state": record.state.value,
        "age_s": now - record.created_at,
        "created_wall": _BOOT_WALL + record.created_at,
        "error": record.error,
        "has_table": record.table_parquet is not None,
        "parquet_bytes": summary.get("parquet_bytes"),
        "summary": summary or None,
        "quantum": result.get("quantum"),
        "prepare_timings": record.prepare_timings,
        "push_timings": record.push_timings,
        "table_url": f"/result/{job_id}/table",
    }


@app.get("/admin/exposures", dependencies=[Depends(check_auth)])
async def admin_exposures(response: Response, limit: int = JOBS_PAGE_MAX):
    """The viewable exposures, newest first: just an id and a visit each.

    Exists so the results page's exposure arrows can find their neighbours. It is
    deliberately *not* /admin/jobs with a filter: a row there carries the whole
    summary, both timing dicts and the quantum, which is two orders of magnitude
    more bytes than this page needs to label two buttons -- and that endpoint's
    `limit` is the dashboard's table length, so reusing it would silently make how
    far an operator can step through exposures depend on an unrelated input box.

    Only jobs that still hold a table are listed, because those are the only ones
    /results can render -- so stepping never lands on a page that 409s. The
    front-end's own retention is therefore what bounds this, which is the honest
    answer: a job whose parquet is gone is not an exposure you can look at.
    """
    limit = max(1, min(JOBS_PAGE_MAX, limit))
    out = []
    # Newest first and stopping at `limit`, like /admin/jobs -- but counting only
    # the rows that survive the filter, so a long tail of ERRORs cannot crowd out
    # the viewable exposures the arrows are for. The cost of that is O(len(JOBS))
    # rather than O(limit) when fewer than `limit` jobs have tables at all, which
    # is a dict walk with no parquet read in it, on a route that runs when an
    # operator presses an arrow rather than on any poll loop.
    for job_id, record in reversed(JOBS.items()):
        if record.table_parquet is None:
            continue
        quantum = (record.result or {}).get("quantum") or {}
        out.append({"job_id": job_id, "visit": quantum.get("visit")})
        if len(out) >= limit:
            break

    response.headers.update(_NO_STORE)
    return {"exposures": out, "returned": len(out), "limit": limit}


@app.get("/admin/jobs", dependencies=[Depends(check_auth)])
async def admin_jobs(response: Response, limit: int = JOBS_PAGE_DEFAULT):
    """The job list, newest first. The one thing no existing endpoint can produce:
    /health carries counts but no ids, and /status and /result both need an id
    you already have.

    O(limit), not O(len(JOBS)): JOBS is insertion-ordered, so the newest rows are
    the tail and islice(reversed(...)) stops after `limit` of them. Iterating the
    live view is safe here because this coroutine never awaits, and the only other
    writer -- FlightPool._fail_jobs on the reader thread -- mutates records rather
    than the dict.
    """
    limit = max(1, min(JOBS_PAGE_MAX, limit))
    now = time.monotonic()
    items = itertools.islice(reversed(JOBS.items()), limit)
    rows = [_job_row(job_id, record, now) for job_id, record in items]

    # Plural, and unbounded in principle: one job per flight can be non-terminal
    # at once, plus any number merely PREPARED and waiting to be pushed.
    active = [row["job_id"] for row in rows if row["state"] in _ACTIVE_STATES]

    response.headers.update(_NO_STORE)
    # `counts` duplicates /health's `jobs` on purpose: the table and the per-node
    # badges have to come from one snapshot or they visibly disagree mid-poll.
    return {
        "now_wall": time.time(),
        "total": len(JOBS),
        "returned": len(rows),
        "limit": limit,
        "counts": _job_counts(),
        "active_job_ids": active,
        "jobs": rows,
    }


# Both log routes are sync `def`, unlike every other route here: Starlette runs a
# non-async route in its threadpool, which keeps this file I/O off the event loop
# that is also servicing a 308 MB push.
@app.get("/admin/log", dependencies=[Depends(check_auth)])
def admin_log(
    response: Response,
    bytes: int = logtail.DEFAULT_TAIL_BYTES,
    since_size: int = -1,
):
    """A bounded tail of the pipeline log.

    `since_size` lets an idle 2 s poll cost ~40 bytes instead of 64 KB. Size is
    the only cheap change detector available: the file is append-only.
    """
    response.headers.update(_NO_STORE)
    if since_size >= 0:
        try:
            size = os.stat(logtail.log_path()).st_size
        except FileNotFoundError:
            size = -1
        if size == since_size:
            return {"unchanged": True, "exists": True, "size": size}
    return logtail.tail(bytes)


@app.get("/admin/result/{job_id}/rows", dependencies=[Depends(check_auth)])
def admin_result_rows(
    response: Response,
    job_id: str,
    offset: int = 0,
    limit: int = table_view.PAGE_DEFAULT,
    columns: str = "",
):
    """The wavefront table as JSON, for the results page.

    Sync `def` like the log routes, and for a stronger reason: decoding parquet is
    CPU-bound, so on the event loop it would stall the same loop that streams a
    308 MB push. Starlette runs a non-async route in its threadpool.

    `/result/{job_id}/table` stays the way to get the bytes themselves -- this is a
    view of them, and deliberately not a second copy of the download.
    """
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")
    if record.state == JobState.ERROR:
        raise HTTPException(409, record.error or "job failed")
    if record.table_parquet is None:
        raise HTTPException(409, f"no table for job in state {record.state}")

    wanted = [c for c in columns.split(",") if c] or None
    body = table_view.read(record.table_parquet, offset=offset, limit=limit,
                           columns=wanted)
    body["job_id"] = job_id
    body["summary"] = (record.result or {}).get("summary")
    body["quantum"] = (record.result or {}).get("quantum")
    response.headers.update(_NO_STORE)
    return body


@app.get("/admin/log/{job_id}", dependencies=[Depends(check_auth)])
def admin_log_job(response: Response, job_id: str):
    """The log lines one job produced.

    A job with no slice answers 200 with found=false rather than 404: any job
    older than the search window legitimately has none, and that is not an error
    about the job.
    """
    response.headers.update(_NO_STORE)
    return logtail.job_slice(job_id)
