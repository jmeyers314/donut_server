"""Coordinator liveness, restart and error-surface tests.

Deliberately imports no part of the LSST stack, at module scope or anywhere else:
Coord is driven against `tests/fake_coordinator`, injected through Coord's ctor
seams, so this module runs anywhere and needs no calibs.

pytest_asyncio is not installed, so every test is a sync function that drives one
async body through asyncio.run().
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from contextlib import asynccontextmanager
from multiprocessing import shared_memory
from multiprocessing.connection import wait as mp_wait

import pytest
from fastapi.testclient import TestClient

import fake_coordinator
from lsst.ts.donut_server import protocol
from lsst.ts.donut_server import server
from lsst.ts.donut_server.server import (
    Coord,
    CoordinatorLost,
    CoordinatorUnavailable,
    CoordState,
)

# Small enough that a test never touches the 513 MB production block.
TEST_SHM_SIZE = 4096

# The fake only needs to be spawned, not to import afw + ts_wep, so bring-up is
# well under a second; these bounds are loose enough for a loaded machine.
READY_TIMEOUT_S = 30.0
DETECT_TIMEOUT_S = 5.0


def make_coord(script=(), **kwargs):
    kwargs.setdefault("backoff", (0.0, 0.0, 0.0))
    return Coord(
        target=fake_coordinator.main,
        target_args=(list(script),),
        shm_size=TEST_SHM_SIZE,
        **kwargs,
    )


@asynccontextmanager
async def started(coord):
    await coord.start()
    try:
        yield coord
    finally:
        await coord.aclose()


async def wait_for(predicate, timeout=READY_TIMEOUT_S, what="condition"):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.02)


async def wait_ready(coord, timeout=READY_TIMEOUT_S):
    await wait_for(lambda: coord.state is CoordState.READY, timeout, "READY")


async def wait_generation(coord, generation, timeout=READY_TIMEOUT_S):
    await wait_for(
        lambda: coord.generation >= generation and coord.state is CoordState.READY,
        timeout,
        f"generation {generation}",
    )


# --------------------------------------------------------------- 1. the seam


def test_fake_child_is_spawnable_and_says_hello():
    """Validates the seam itself before anything depends on it: the fake is
    importable in a spawned child and completes the hello handshake."""

    async def body():
        async with started(make_coord()) as coord:
            await wait_ready(coord)
            assert coord.generation == 1
            assert coord.spawns == 1
            assert coord.child_pid is not None
            resp = await coord.send_command({"cmd": "ping", "echo": "hi"})
            assert resp["ok"] is True
            assert resp["echo"] == "hi"

    asyncio.run(body())


# ------------------------------------------- 2. the parent closes the child end


def test_a_lone_dead_child_does_let_the_pipe_reach_eof():
    """Baseline for the test below: with no descendants holding a copy of the
    child end, EOF *is* reachable and a plain recv() would raise."""

    async def body():
        async with started(make_coord()) as coord:
            await wait_ready(coord)
            conn = coord._conn
            os.kill(coord.child_pid, signal.SIGKILL)
            assert conn.poll(DETECT_TIMEOUT_S) is True, "pipe never became readable"
            with pytest.raises(EOFError):
                conn.recv()

    asyncio.run(body())


def test_a_forked_grandchild_makes_both_fd_wakeup_sources_useless():
    """The measured reason _await_reply must poll proc.is_alive().

    The coordinator's fork workers inherit copies of both the pipe fd and the spawn
    sentinel's write end. So once the coordinator dies with a worker still up,
    neither fd-based source reports it: the sentinel stays unreadable, and the pipe
    never reaches EOF either -- even though the parent correctly closed its own copy
    of the child end. A plain recv() blocks indefinitely right here. waitpid(WNOHANG)
    is the only source that tells the truth, which is what makes the poll loop the
    actual fix for the wedge.
    """

    async def body():
        async with started(make_coord(["fork_holder"])) as coord:
            await wait_ready(coord)
            conn, proc = coord._conn, coord._proc
            # Provoke the crash directly rather than through send_command, so this
            # test observes the raw fds instead of the detection built on them.
            conn.send({"cmd": "ping"})
            await wait_for(lambda: not proc.is_alive(), 10.0, "the child to exit")

            assert mp_wait([proc.sentinel], timeout=1.0) == [], "sentinel was readable"
            assert conn.poll(1.0) is False, "pipe reached EOF"

    asyncio.run(body())


# ------------------------------------------------------ 3-4. prompt detection


def test_dead_child_is_detected_not_hung():
    """The regression test for the permanent-deadlock bug: a command whose child
    dies must raise promptly rather than block forever under _lock."""

    async def body():
        async with started(make_coord(["crash"])) as coord:
            await wait_ready(coord)
            t0 = time.monotonic()
            with pytest.raises(CoordinatorLost):
                await asyncio.wait_for(
                    coord.send_command({"cmd": "ping"}), timeout=DETECT_TIMEOUT_S * 2
                )
            assert time.monotonic() - t0 < DETECT_TIMEOUT_S
            assert coord.last_loss is not None
            assert coord.last_loss["provoked_by"] == "ping"

    asyncio.run(body())


def test_detection_is_prompt_when_a_grandchild_holds_the_sentinel():
    """Fails if detection relies on the spawn sentinel alone.

    popen_fork closes only the fds it creates, so a forked grandchild inherits a
    copy of the sentinel's write end and holds it unreadable. This is the
    crash-during-runQuantum shape, i.e. the likeliest real crash.
    """

    async def body():
        async with started(make_coord(["fork_holder"])) as coord:
            await wait_ready(coord)
            t0 = time.monotonic()
            with pytest.raises(CoordinatorLost):
                await asyncio.wait_for(
                    coord.send_command({"cmd": "ping"}), timeout=DETECT_TIMEOUT_S * 2
                )
            assert time.monotonic() - t0 < DETECT_TIMEOUT_S

    asyncio.run(body())


def test_truncated_reply_is_a_loss_not_a_hang():
    """half_reply announces 1 KB and sends 16 bytes: the EOFError/OSError branch."""

    async def body():
        async with started(make_coord(["half_reply"])) as coord:
            await wait_ready(coord)
            with pytest.raises(CoordinatorLost):
                await asyncio.wait_for(
                    coord.send_command({"cmd": "ping"}), timeout=DETECT_TIMEOUT_S * 2
                )

    asyncio.run(body())


# ----------------------------------------------------------- 5-8. restart ladder


def test_restart_produces_a_working_child():
    async def body():
        async with started(make_coord(["crash"])) as coord:
            await wait_ready(coord)
            with pytest.raises(CoordinatorLost):
                await coord.send_command({"cmd": "ping"})
            await wait_generation(coord, 2)
            assert coord.spawns == 2
            assert coord.state is CoordState.READY
            resp = await coord.send_command({"cmd": "ping"})
            assert resp["ok"] is True

    asyncio.run(body())


def test_concurrent_losses_cause_exactly_one_restart():
    async def body():
        async with started(make_coord(["crash"])) as coord:
            await wait_ready(coord)
            results = await asyncio.gather(
                *(coord.send_command({"cmd": "ping"}) for _ in range(3)),
                return_exceptions=True,
            )
            lost = [r for r in results if isinstance(r, CoordinatorLost)]
            unavailable = [r for r in results if isinstance(r, CoordinatorUnavailable)]
            assert len(lost) == 1, results
            # The other two arrive mid-restart and are told to come back, rather
            # than being retried against a child that is not there yet.
            assert len(unavailable) == 2, results
            await wait_generation(coord, 2)
            assert coord.spawns == 2

    asyncio.run(body())


def test_bounded_attempts_end_in_degraded():
    async def body():
        coord = Coord(
            target=fake_coordinator.exit_before_hello,
            target_args=([],),
            shm_size=TEST_SHM_SIZE,
            max_restart_attempts=2,
            backoff=(0.0, 0.0),
        )
        async with started(coord):
            await wait_for(
                lambda: coord.state is CoordState.DEGRADED, READY_TIMEOUT_S, "DEGRADED"
            )
            assert coord.spawns == 2
            assert coord.degraded_reason
            with pytest.raises(CoordinatorUnavailable):
                await coord.send_command({"cmd": "ping"})
            # Terminal: no further spawns without POST /admin/restart.
            assert coord.spawns == 2

    asyncio.run(body())


def test_a_successful_hello_resets_the_failure_counter():
    """So that unrelated crashes hours apart never accumulate into DEGRADED."""

    async def body():
        async with started(make_coord(["crash"])) as coord:
            await wait_ready(coord)
            assert coord.restart_attempts == 0
            with pytest.raises(CoordinatorLost):
                await coord.send_command({"cmd": "ping"})
            await wait_generation(coord, 2)
            assert coord.restart_attempts == 0

    asyncio.run(body())


# ------------------------------------------------------------- 9. re-priming


def test_last_prepare_is_replayed_after_a_restart():
    async def body():
        # Command 1 is the prepare (fine); command 2 crashes the child.
        async with started(make_coord([None, "crash"])) as coord:
            await wait_ready(coord)
            resp = await coord.send_command({"cmd": "prepare", "band": "r"})
            assert resp["seen"] == 1
            assert coord.primed_args == {"cmd": "prepare", "band": "r"}

            with pytest.raises(CoordinatorLost):
                await coord.send_command({"cmd": "ping"})
            await wait_generation(coord, 2)

            # The replacement child has already handled the replayed prepare, so
            # this command is its second -- no /prepare from the producer.
            resp = await coord.send_command({"cmd": "ping"})
            assert resp["seen"] == 2
            assert coord.primed_args == {"cmd": "prepare", "band": "r"}

    asyncio.run(body())


def test_config_overrides_are_replayed_verbatim_and_refresh_the_snapshot():
    """The whole reason overrides ride on /prepare rather than their own endpoint.

    Two things must hold after a restart. The override list -- including the full -C
    body, which no echo ever carries -- has to be replayed, or the replacement child
    would silently run defaults while /health still claimed it was primed. And the
    cached config snapshot has to advance to the new generation, which is the part
    that was structurally easy to get wrong: _reprime talks to the child through
    _exchange directly, bypassing send_command, so caching done only there would
    never run on this path.
    """
    prepare = {
        "cmd": "prepare",
        "band": "r",
        "config_overrides": [
            {"kind": "value", "field": "maxFitScatter", "value": "2.0"},
            {"kind": "python", "name": "/home/op/tweaks.py", "text": "config.savePlots = True\n"},
        ],
    }

    async def body():
        # Command 1 is the prepare (fine); command 2 crashes the child.
        async with started(make_coord([None, "crash"])) as coord:
            await wait_ready(coord)
            await coord.send_command(dict(prepare))
            first = coord.config_snapshot
            assert first is not None and first.generation == 1

            with pytest.raises(CoordinatorLost):
                await coord.send_command({"cmd": "ping"})
            await wait_generation(coord, 2)

            # Replayed verbatim: the -C body survives the round trip intact.
            assert coord.primed_args == prepare
            # ...and the snapshot now describes the *new* child, so /config does not
            # report stale for a coordinator that is in fact correctly primed.
            assert coord.config_snapshot.generation == 2
            assert coord.config_snapshot.dump != first.dump  # fake keys it on pid
            # The digest is retained for /health, still without the body.
            assert coord.config_snapshot.overrides[1]["name"] == "/home/op/tweaks.py"
            assert "text" not in coord.config_snapshot.overrides[1]

    asyncio.run(body())


def test_prepare_is_not_replayed_when_a_prepare_provoked_the_crash():
    """Anti-crash-loop: replaying the command that killed the last child would
    very likely kill this one too."""

    async def body():
        # Command 1 (a prepare) succeeds and is cached; command 2 (also a
        # prepare) crashes the child.
        async with started(make_coord([None, "crash"])) as coord:
            await wait_ready(coord)
            await coord.send_command({"cmd": "prepare", "band": "r"})
            with pytest.raises(CoordinatorLost):
                await coord.send_command({"cmd": "prepare", "band": "g"})
            await wait_generation(coord, 2)

            resp = await coord.send_command({"cmd": "ping"})
            assert resp["seen"] == 1, "prepare was replayed despite provoking the crash"

    asyncio.run(body())


# ---------------------------------------------------------------- 10. fd hygiene


def test_no_fd_leak_across_restarts():
    """Pins proc.close() and both Connection.close()s."""

    def fd_count():
        return len(os.listdir("/dev/fd"))

    async def body():
        async with started(make_coord()) as coord:
            await wait_ready(coord)
            baseline = fd_count()
            for _ in range(5):
                generation = coord.generation
                coord.request_restart("fd leak probe")
                await wait_generation(coord, generation + 1)
            assert coord.spawns == 6
            assert abs(fd_count() - baseline) <= 2

    asyncio.run(body())


# --------------------------------------------------- 11-12. wedged but alive


def test_aclose_is_prompt_with_a_hanging_child():
    async def body():
        coord = make_coord(["hang"])
        await coord.start()
        await wait_ready(coord)
        name = coord.shm.name
        stuck = asyncio.create_task(coord.send_command({"cmd": "ping"}))
        await asyncio.sleep(0.5)
        assert coord.in_flight == 1

        t0 = time.monotonic()
        await coord.aclose()
        assert time.monotonic() - t0 < 10.0

        with pytest.raises(CoordinatorLost):
            await stuck
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    asyncio.run(body())


def test_admin_restart_recovers_a_hanging_child():
    async def body():
        async with started(make_coord(["hang"])) as coord:
            await wait_ready(coord)
            wedged_pid = coord.child_pid
            stuck = asyncio.create_task(coord.send_command({"cmd": "ping"}))
            await asyncio.sleep(0.5)

            # is_alive() cannot see a wedged-but-alive child, by design; this is
            # the operator's handle for it.
            assert coord.is_alive() is True
            assert coord.request_restart("wedged") == wedged_pid

            with pytest.raises(CoordinatorLost):
                await stuck
            await wait_generation(coord, 2)
            resp = await coord.send_command({"cmd": "ping"})
            assert resp["ok"] is True

    asyncio.run(body())


def test_admin_restart_is_the_way_out_of_degraded():
    async def body():
        coord = Coord(
            target=fake_coordinator.exit_before_hello,
            target_args=([],),
            shm_size=TEST_SHM_SIZE,
            max_restart_attempts=1,
            backoff=(0.0,),
        )
        async with started(coord):
            await wait_for(
                lambda: coord.state is CoordState.DEGRADED, READY_TIMEOUT_S, "DEGRADED"
            )
            # A restart can only help if the target is fixed, which for the fake
            # means swapping it -- the point here is that DEGRADED is escapable.
            coord._target = fake_coordinator.main
            coord.request_restart("operator")
            await wait_generation(coord, 1)
            assert coord.state is CoordState.READY

    asyncio.run(body())


# ------------------------------------------------------------- 13. endpoints

TOKEN = "test-token"


def make_pool(*coords):
    """A FlightPool wrapping already-built Coords, bypassing its own construction.

    The endpoint tests below each want a specific misbehaving child, which is
    expressed through Coord's ctor seams -- so the pool is assembled around the
    Coords rather than asked to build them.
    """
    pool = server.FlightPool.__new__(server.FlightPool)
    pool.flights = list(coords)
    for i, coord in enumerate(coords):
        coord.index = i
        coord._on_restart = pool._loss_handler(i)
    pool._by_job = {}
    pool._last_used = [0.0] * len(coords)
    pool._queued = [0] * len(coords)
    return pool


def make_client(monkeypatch, *coords):
    monkeypatch.setenv("DONUT_SERVER_TOKEN", TOKEN)
    # lifespan reads the module global, so this must be patched before entering.
    monkeypatch.setattr(server, "pool", make_pool(*coords))
    monkeypatch.setattr(server, "JOBS", {})
    return TestClient(server.app)


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def poll_until(predicate, timeout=READY_TIMEOUT_S, what="condition"):
    """Sync spin, for tests driving the app through TestClient: its event loop runs
    on another thread, so sleeping here does not block bring-up."""
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def test_health_is_reachable_without_a_token(monkeypatch):
    coord = make_coord()
    with make_client(monkeypatch, coord) as client:
        # /health legitimately 503s while the child is still starting, so wait for
        # the ready verdict rather than racing bring-up.
        poll_until(lambda: coord.state is CoordState.READY, what="READY")
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready", "ready": True}
        assert resp.headers["cache-control"] == "no-store"

        detail = client.get("/health", headers=auth()).json()
        assert detail["n_flights"] == 1
        assert detail["n_ready"] == 1
        (flight,) = detail["flights"]
        assert flight["generation"] == 1
        assert flight["alive"] is True
        assert flight["primed"] is False


def test_health_503s_when_degraded_and_reports_the_exitcode(monkeypatch):
    coord = Coord(
        target=fake_coordinator.exit_before_hello,
        target_args=([],),
        shm_size=TEST_SHM_SIZE,
        max_restart_attempts=1,
        backoff=(0.0,),
    )
    with make_client(monkeypatch, coord) as client:
        poll_until(lambda: coord.state is CoordState.DEGRADED, what="DEGRADED")

        resp = client.get("/health", headers=auth())
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["ready"] is False
        (flight,) = body["flights"]
        assert flight["degraded_reason"]
        # exitcode is the highest-value field in last_loss: 3 here because
        # exit_before_hello calls os._exit(3).
        assert flight["last_loss"]["exitcode"] == 3


def test_prepare_503s_with_a_reason_when_the_coordinator_is_degraded(monkeypatch):
    coord = Coord(
        target=fake_coordinator.exit_before_hello,
        target_args=([],),
        shm_size=TEST_SHM_SIZE,
        max_restart_attempts=1,
        backoff=(0.0,),
    )
    with make_client(monkeypatch, coord) as client:
        poll_until(lambda: coord.state is CoordState.DEGRADED, what="DEGRADED")

        resp = client.post(
            "/prepare",
            json={"band": "r", "boresight_ra": 1.0, "boresight_dec": 2.0},
            headers=auth(),
        )
        assert resp.status_code == 503
        assert resp.json()["reason"] == "coordinator_degraded"
        assert resp.headers["retry-after"] == "3600"


def test_push_503s_and_marks_the_job_errored(monkeypatch):
    # Command 1 is the prepare; command 2 (the push) crashes the child.
    coord = make_coord([None, "crash"])
    with make_client(monkeypatch, coord) as client:
        job_id = client.post(
            "/prepare",
            json={"band": "r", "boresight_ra": 1.0, "boresight_dec": 2.0},
            headers=auth(),
        ).json()["job_id"]

        blob = protocol.pack_blob({"R00_SW0:img": b"pixels", "R00_SW0:meta": b"meta"})
        resp = client.post(f"/push/{job_id}", content=blob, headers=auth())
        assert resp.status_code == 503
        assert resp.json()["reason"] == "coordinator_lost"
        assert resp.headers["retry-after"] == "30"

        # /status and /result must agree with what the client just saw, rather
        # than long-polling a job that can never finish.
        assert client.get(f"/status/{job_id}", headers=auth()).json()["state"] == "ERROR"
        result = client.get(f"/result/{job_id}", headers=auth()).json()
        assert result["ready"] is True
        assert result["state"] == "ERROR"


def test_health_503s_while_starting(monkeypatch):
    """A child partway through its imports is alive but cannot answer anything, so
    /health must not call it ready. is_alive() alone would."""
    coord = Coord(
        target=fake_coordinator.hang_before_hello,
        target_args=([],),
        shm_size=TEST_SHM_SIZE,
    )
    with make_client(monkeypatch, coord) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"status": "starting", "ready": False}
        assert coord.is_alive() is True


# ------------------------------------------------------------- 14. the pool

def prepare(client, band="r"):
    return client.post(
        "/prepare",
        json={"band": band, "boresight_ra": 1.0, "boresight_dec": 2.0},
        headers=auth(),
    )


def push_blob():
    return protocol.pack_blob({"R00_SW0:img": b"pixels", "R00_SW0:meta": b"meta"})


def test_each_flight_gets_its_own_child():
    """The whole basis of the pool: N Coords are N processes with N shared blocks,
    not N views onto one. Distinct pids is the cheapest proof."""

    async def body():
        pool = server.FlightPool(
            2,
            target=fake_coordinator.main,
            target_args=([],),
            shm_size=TEST_SHM_SIZE,
            backoff=(0.0, 0.0, 0.0),
        )
        await pool.start()
        try:
            for coord in pool.flights:
                await wait_ready(coord)
            pids = {c.child_pid for c in pool.flights}
            blocks = {c.shm.name for c in pool.flights}
            assert len(pids) == 2
            assert len(blocks) == 2
            assert [c.index for c in pool.flights] == [0, 1]
        finally:
            await pool.aclose()

    asyncio.run(body())


def test_a_loss_on_one_flight_does_not_error_another_flights_job(monkeypatch):
    """The correctness fix the pool needs most. A single-coordinator server could
    safely fail every non-terminal job on a loss; with two lanes that would kill a
    healthy job on the other one, for no reason its producer could diagnose."""
    # Flight 0's second command (its push) crashes the child; flight 1 is healthy.
    with make_client(monkeypatch, make_coord([None, "crash"]), make_coord()) as client:
        for coord in server.pool.flights:
            poll_until(lambda c=coord: c.state is CoordState.READY, what="READY")

        # One job per flight. Distinct bands, so the second prepare cannot take
        # select()'s already-primed branch back onto flight 0; LRU then sends it to
        # flight 1, which is asserted rather than assumed.
        a = prepare(client, "r").json()
        b = prepare(client, "g").json()
        assert (a["flight"], b["flight"]) == (0, 1)

        # Strand b mid-flight by hand: driving a real concurrent push through
        # TestClient would need two threads, and what is under test here is the
        # filtering in _fail_jobs, not the concurrency.
        server.JOBS[b["job_id"]].state = server.JobState.COMPUTING

        resp = client.post(f"/push/{a['job_id']}", content=push_blob(), headers=auth())
        assert resp.status_code == 503
        assert resp.json()["flight"] == 0

        assert server.JOBS[a["job_id"]].state is server.JobState.ERROR
        assert server.JOBS[b["job_id"]].state is server.JobState.COMPUTING


def test_a_repeated_prepare_stays_on_the_flight_that_is_already_primed(monkeypatch):
    """Case 1 of select(): reusing a primed flight is free, and warming the second
    lane with an identical config would only evict something useful from it."""
    with make_client(monkeypatch, make_coord(), make_coord()) as client:
        for coord in server.pool.flights:
            poll_until(lambda c=coord: c.state is CoordState.READY, what="READY")

        first = prepare(client, "r").json()["flight"]
        # A different config goes to the other lane; the original then comes back to
        # the lane still holding it, rather than to the now-LRU one.
        other = prepare(client, "g").json()["flight"]
        again = prepare(client, "r").json()["flight"]

        assert other != first
        assert again == first


def test_a_push_for_an_unrouted_job_falls_back_rather_than_404ing(monkeypatch):
    """The orphaned-push case: the job record exists but the routing entry does
    not. Any flight can serve it -- the push carries prepared_key, which the
    coordinator uses to reload that job's config -- so this must not be an error."""
    with make_client(monkeypatch, make_coord()) as client:
        job_id = prepare(client).json()["job_id"]
        server.pool._by_job.clear()

        resp = client.post(f"/push/{job_id}", content=push_blob(), headers=auth())

        # The fake has no real pipeline, so the interesting assertion is that this
        # reached a flight at all rather than 404ing on the missing route.
        assert resp.status_code != 404
        assert server.pool._by_job[job_id] == 0


def test_health_is_partial_when_one_flight_is_down(monkeypatch):
    """200 with a warning, not 503: a pool with one healthy lane serves every
    request correctly, and a supervisor must not kill it for running slower."""
    broken = Coord(
        target=fake_coordinator.exit_before_hello,
        target_args=([],),
        shm_size=TEST_SHM_SIZE,
        max_restart_attempts=1,
        backoff=(0.0,),
    )
    with make_client(monkeypatch, make_coord(), broken) as client:
        poll_until(lambda: server.pool.flights[0].state is CoordState.READY, what="READY")
        poll_until(lambda: broken.state is CoordState.DEGRADED, what="DEGRADED")

        resp = client.get("/health", headers=auth())
        body = resp.json()

        assert resp.status_code == 200
        assert body["status"] == "partial"
        assert body["ready"] is True
        assert (body["n_flights"], body["n_ready"]) == (2, 1)
        assert body["flights"][1]["status"] == "degraded"

        # And a prepare still succeeds, on the lane that is up.
        assert prepare(client).json()["flight"] == 0


def test_restart_targets_one_flight_and_validates_the_index(monkeypatch):
    with make_client(monkeypatch, make_coord(), make_coord()) as client:
        for coord in server.pool.flights:
            poll_until(lambda c=coord: c.state is CoordState.READY, what="READY")
        untouched = server.pool.flights[1].child_pid

        body = client.post("/admin/restart?flight=0", headers=auth()).json()
        assert [f["index"] for f in body["flights"]] == [0]

        poll_until(lambda: server.pool.flights[0].generation >= 2, what="generation 2")
        # The other lane keeps its child, and its warm caches with it.
        assert server.pool.flights[1].child_pid == untouched
        assert server.pool.flights[1].generation == 1

        assert client.post("/admin/restart?flight=9", headers=auth()).status_code == 400

        # Bare still means every flight, which is what it meant with one.
        body = client.post("/admin/restart", headers=auth()).json()
        assert [f["index"] for f in body["flights"]] == [0, 1]


def test_two_flights_compute_concurrently():
    """The claim the whole change exists to make. Two 1 s commands issued together
    must finish in ~1 s, not ~2: with a single coordinator the second queues behind
    the first on the one _lock, and this is the only test that can tell the
    difference between the two designs."""

    async def body():
        pool = server.FlightPool(
            2,
            target=fake_coordinator.main,
            target_args=(["slow:1.0"],),
            shm_size=TEST_SHM_SIZE,
            backoff=(0.0, 0.0, 0.0),
        )
        await pool.start()
        try:
            for coord in pool.flights:
                await wait_ready(coord)
            t0 = time.monotonic()
            await asyncio.gather(
                *(c.send_command({"cmd": "ping"}) for c in pool.flights)
            )
            elapsed = time.monotonic() - t0
            assert 1.0 <= elapsed < 1.8, elapsed
        finally:
            await pool.aclose()

    asyncio.run(body())


def test_one_flight_reproduces_single_coordinator_serialization():
    """The safety net for the whole change: N=1 must behave exactly as before, which
    for the pool's purposes means one lane, one lock, and no concurrency."""

    async def body():
        pool = server.FlightPool(
            1,
            target=fake_coordinator.main,
            target_args=(["slow:0.5", "slow:0.5"],),
            shm_size=TEST_SHM_SIZE,
            backoff=(0.0, 0.0, 0.0),
        )
        await pool.start()
        try:
            (coord,) = pool.flights
            await wait_ready(coord)
            t0 = time.monotonic()
            await asyncio.gather(
                coord.send_command({"cmd": "ping"}),
                coord.send_command({"cmd": "ping"}),
            )
            # Serialized behind the one _lock, as it always was.
            assert time.monotonic() - t0 >= 1.0
            # And every prepare and push necessarily routes to the same lane.
            assert pool.select({"cmd": "prepare"}).index == 0
            assert pool.for_push("never-prepared").index == 0
        finally:
            await pool.aclose()

    asyncio.run(body())


def test_num_flights_comes_from_the_environment(monkeypatch):
    """The transport from `donutServer.py --num-flights` is the env var, because
    uvicorn imports this module in-process."""
    monkeypatch.delenv("DONUT_SERVER_NUM_FLIGHTS", raising=False)
    assert server.num_flights() == 2
    monkeypatch.setenv("DONUT_SERVER_NUM_FLIGHTS", "4")
    assert server.num_flights() == 4
    # Neither a typo nor a nonsense value may leave the service with no lanes.
    monkeypatch.setenv("DONUT_SERVER_NUM_FLIGHTS", "0")
    assert server.num_flights() == 1
    monkeypatch.setenv("DONUT_SERVER_NUM_FLIGHTS", "two")
    assert server.num_flights() == 2
