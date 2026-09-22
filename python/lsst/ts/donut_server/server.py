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
import ipaddress
import itertools
import math
import multiprocessing as mp
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import shared_memory
from multiprocessing.connection import wait as mp_wait
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from lsst.ts.donut_server import logtail
from lsst.ts.donut_server import protocol
from lsst.ts.donut_server import table_view

MAX_PUSH_BYTES = 512 * 1024 * 1024  # early size-cap rejection

# The block also holds the header and descriptors, which sit ahead of the
# payload that MAX_PUSH_BYTES caps.
SHM_SIZE = MAX_PUSH_BYTES + 1024 * 1024

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


def _fail_in_flight_jobs(loss: dict) -> None:
    """Fail every job that was mid-flight when the coordinator died.

    Without this a record stranded in COMPUTING makes /result?wait=N long-poll a
    job that can never finish.
    """
    for record in JOBS.values():
        if record.state in (JobState.RECEIVING, JobState.COMPUTING):
            was = record.state
            record.state = JobState.ERROR
            record.error = f"coordinator lost during {was.value}: {loss.get('reason')}"


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
    ) -> None:
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
        self._child_pid: Optional[int] = None
        self._child_started_at: Optional[float] = None
        self._last_loss: Optional[dict] = None
        self._degraded_reason: Optional[str] = None
        self._primed_args: Optional[dict] = None
        self._loss_provoked_by: Optional[str] = None
        self._reprime_error: Optional[str] = None

    # ---------------------------------------------------------------- accessors

    @property
    def shm(self):
        return self._shm

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
    def child_pid(self) -> Optional[int]:
        return self._child_pid

    @property
    def last_loss(self) -> Optional[dict]:
        return self._last_loss

    @property
    def degraded_reason(self) -> Optional[str]:
        return self._degraded_reason

    @property
    def primed_args(self) -> Optional[dict]:
        return self._primed_args

    @property
    def in_flight(self) -> int:
        return self._in_flight

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
        self._proc = ctx.Process(
            target=target, args=(child_conn, self._shm.name) + self._target_args
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
        if not resp.get("ok"):
            # Auto-re-priming is an availability optimization, not a correctness
            # requirement: run_job cross-checks the raws' band against the loaded
            # calibs, and the refcat coverage check is equivalent, so an unprimed
            # coordinator fails the next push loudly rather than producing bad
            # wavefronts. Leave the child READY but unprimed.
            self._primed_args = None
            self._reprime_error = resp.get("error", "prepare replay failed")

    def _note_loss(self, exc: Exception, provoked_by: Optional[str] = None) -> None:
        proc = self._proc
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
            await self._ensure_usable()
            try:
                resp = await self._exchange(command)
            except CoordinatorLost as exc:
                self._note_loss(exc, provoked_by=command.get("cmd"))
                # Set under _lock, because requests arriving mid-restart must be
                # able to check state *without* the lock -- _restart holds it.
                self._start_restart()
                # The failed command is deliberately not retried: a push that
                # killed the coordinator may kill it again, and the job's state
                # machine has already moved on. 503 and let the producer decide.
                raise
            if command.get("cmd") == "prepare" and resp.get("ok"):
                self._primed_args = dict(command)
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


coord = Coord(on_restart=_fail_in_flight_jobs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await coord.start()
    yield
    await coord.aclose()


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
    if isinstance(exc, CoordinatorLost):
        reason = "coordinator_lost"
    elif coord.state is CoordState.DEGRADED:
        reason = "coordinator_degraded"
    else:
        reason = "coordinator_restarting"
    retry_after = _RETRY_AFTER[reason]
    return JSONResponse(
        status_code=503,
        content={
            "reason": reason,
            "error": str(exc),
            "state": coord.state.value,
            "generation": coord.generation,
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


@app.post("/prepare", dependencies=[Depends(check_auth)])
async def prepare(body: dict):
    # The boresight is required: it is what lets the coordinator pre-load
    # reference catalog shards for the pointing before any pixels exist,
    # keeping ~450 ms of shard resharding off the push critical path.
    boresight_ra = _required_float(body, "boresight_ra")
    boresight_dec = _required_float(body, "boresight_dec")

    job_id = str(uuid.uuid4())
    resp = await coord.send_command(
        {
            "cmd": "prepare",
            "band": body.get("band"),
            "calib_selector": body.get("calib_selector"),
            "boresight_ra": boresight_ra,
            "boresight_dec": boresight_dec,
        }
    )
    if not resp.get("ok"):
        JOBS[job_id] = JobRecord(job_id=job_id, state=JobState.ERROR, error=resp.get("error"))
        raise HTTPException(500, resp.get("error", "prepare failed"))

    JOBS[job_id] = JobRecord(job_id=job_id, state=JobState.PREPARED, prepare_timings=resp.get("timings"))
    # `generation` lets the producer's own logs attribute a latency spike to a
    # coordinator restart.
    return {
        "job_id": job_id,
        "state": JobState.PREPARED,
        "timings": resp.get("timings"),
        "generation": coord.generation,
    }


@app.post("/push/{job_id}", dependencies=[Depends(check_auth)])
async def push(job_id: str, request: Request):
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "unknown job_id")
    if record.state != JobState.PREPARED:
        raise HTTPException(409, f"job in state {record.state}, expected PREPARED")

    async with coord.push_lock:
        view = memoryview(coord.shm.buf)
        try:
            return await _receive_and_dispatch(job_id, record, request, view)
        finally:
            # Must be released before Coord.stop() can close the block.
            view.release()


async def _receive_and_dispatch(job_id, record, request: Request, view: memoryview) -> dict:
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
        resp = await coord.send_command({"cmd": "push", "job_id": job_id, "layout": layout})
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
    return {
        "job_id": job_id,
        "state": record.state,
        "timings": timings,
        "generation": coord.generation,
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


def _job_counts() -> dict:
    counts: dict[str, int] = {}
    for record in JOBS.values():
        counts[record.state.value] = counts.get(record.state.value, 0) + 1
    return counts


@app.get("/health")
async def health(request: Request, response: Response, authorization: str = Header(None)):
    """The status *code* carries the verdict: 200 only when READY, else 503.

    Nothing can supervise an endpoint that always returns 200. Auth is optional
    rather than absent -- a bare remote probe gets the verdict plus a minimal body,
    while a local caller or a valid token gets the full detail.
    """
    ready = coord.state is CoordState.READY
    response.status_code = 200 if ready else 503
    response.headers["Cache-Control"] = "no-store"

    body: dict[str, Any] = {"status": coord.state.value, "ready": ready}
    if not _is_loopback(request):
        token = os.environ.get("DONUT_SERVER_TOKEN")
        if not token or authorization != f"Bearer {token}":
            return body

    body.update(
        {
            "alive": coord.is_alive(),
            "primed": coord.primed_args is not None,
            "primed_args": coord.primed_args,
            "generation": coord.generation,
            "spawns": coord.spawns,
            "restart_attempts": coord.restart_attempts,
            "pid": coord.child_pid,
            "child_uptime_s": coord.child_uptime_s,
            # last_loss.exitcode is the highest-value field here: -11 is a segfault
            # (a fork-safety regression), -9 is SIGKILL, which on a service that
            # oscillates 4-6 GB RSS should be read as macOS jetsam, and 1 is a
            # Python exception with a traceback in the log.
            "last_loss": coord.last_loss,
            "degraded_reason": coord.degraded_reason,
            # alive without ready and with in_flight > 0, held for minutes, is the
            # wedged-but-alive signature that /admin/restart exists for.
            "in_flight": coord.in_flight,
            "jobs": _job_counts(),
        }
    )
    return body


@app.post("/admin/restart", dependencies=[Depends(check_auth)])
async def admin_restart(response: Response):
    """Recover a wedged-but-alive coordinator, and the only way out of DEGRADED.

    There is deliberately no automatic command timeout: a generous bound cannot be
    sized safely against a 6.1-8.9 s push, and a mis-sized one kills healthy work.
    An operator (or an external watchdog) who knows a job has been COMPUTING for
    minutes has strictly more information than a blind timeout.
    """
    pid = coord.request_restart("operator requested restart via /admin/restart")
    response.status_code = 202
    return {"restarting": True, "killed_pid": pid, "state": coord.state.value}


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


@app.get("/admin/jobs", dependencies=[Depends(check_auth)])
async def admin_jobs(response: Response, limit: int = JOBS_PAGE_DEFAULT):
    """The job list, newest first. The one thing no existing endpoint can produce:
    /health carries counts but no ids, and /status and /result both need an id
    you already have.

    O(limit), not O(len(JOBS)): JOBS is insertion-ordered, so the newest rows are
    the tail and islice(reversed(...)) stops after `limit` of them. Iterating the
    live view is safe here because this coroutine never awaits, and the only other
    writer -- _fail_in_flight_jobs on the reader thread -- mutates records rather
    than the dict.
    """
    limit = max(1, min(JOBS_PAGE_MAX, limit))
    now = time.monotonic()
    items = itertools.islice(reversed(JOBS.items()), limit)
    rows = [_job_row(job_id, record, now) for job_id, record in items]

    # At most one job is non-terminal at a time: push_lock serializes pushes and
    # the coordinator's command loop is serial.
    active = next((row["job_id"] for row in rows if row["state"] in _ACTIVE_STATES), None)

    response.headers.update(_NO_STORE)
    # `counts` duplicates /health's `jobs` on purpose: the table and the per-node
    # badges have to come from one snapshot or they visibly disagree mid-poll.
    return {
        "now_wall": time.time(),
        "total": len(JOBS),
        "returned": len(rows),
        "limit": limit,
        "counts": _job_counts(),
        "active_job_id": active,
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
