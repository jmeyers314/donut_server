"""The plot queue, the plot-flight lifecycle, and the routes that serve PNGs.

Imports no part of the LSST stack, like test_dashboard and test_coord_lifecycle:
every PlotFlight here is given a `target` seam pointing at fake_plot_flight, so
nothing imports afw and no figure is ever drawn.

The route tests build `TestClient(server.app)` without the `with` block for the
same reason test_dashboard does -- entering it runs the lifespan, which would
start the module-global real pool.
"""
from __future__ import annotations

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from lsst.ts.donut_server import server
from lsst.ts.donut_server.server import JobRecord, JobState, PlotFlight, PlotQueue

import fake_plot_flight

TOKEN = "test-token"


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


# ------------------------------------------------------------------- the queue

def unstarted(n: int = 1) -> PlotQueue:
    """A queue with `n` flights that are never started.

    The ordering and bookkeeping tests want the data structure, not a child
    process: `start()` is what spawns anything, so constructing the flights is
    free. One flight rather than none because an empty pool deliberately refuses
    to queue -- see test_a_pool_with_no_flights_refuses_to_queue.
    """
    return PlotQueue([PlotFlight(index=i, target=fake_plot_flight.main) for i in range(n)])


def test_the_queue_is_lifo():
    """The whole point of the ordering: when a flight frees up it takes the job
    most recently finished, which is the one an operator is likely looking at."""
    q = unstarted()
    for job in ("a", "b", "c"):
        q.enqueue(job)

    assert [q._take() for _ in range(3)] == ["c", "b", "a"]


def test_enqueue_dedupes():
    q = unstarted()

    assert q.enqueue("a") is True
    assert q.enqueue("a") is False
    assert len(q) == 1


def test_a_pool_with_no_flights_refuses_to_queue():
    """Found in end-to-end testing: without this, `--num-plot-flights 0` still
    accepted the automatic enqueue on every push, and the manifest then reported
    `queued` forever -- so the results page polled for a plot nothing would ever
    draw."""
    q = PlotQueue([])

    assert q.enqueue("a") is False
    assert len(q) == 0
    assert q.state_of("a") is None


def test_a_drawing_job_cannot_be_enqueued_behind_itself():
    """A page polling the manifest must not be able to queue a redraw of the job
    a flight is drawing right now."""
    q = unstarted()
    q._drawing[0] = "a"

    assert q.enqueue("a") is False
    assert q.state_of("a") == "drawing"


def test_take_skips_a_stale_stack_entry():
    """_queued is the authority, not the stack: a job pulled out of the set must
    not come back out of the list."""
    q = unstarted()
    q.enqueue("a")
    q.enqueue("b")
    q._queued.discard("b")

    assert q._take() == "a"
    assert q._take() is None


def test_state_of_reports_queued_then_nothing():
    q = unstarted()
    q.enqueue("a")

    assert q.state_of("a") == "queued"
    q._take()
    assert q.state_of("a") is None


def test_errors_are_capped():
    """Unlike JOBS, which is paged, this is only ever read one job at a time --
    so it needs its own bound or a wedged flight grows it for the life of the
    process."""
    q = unstarted()
    for i in range(server.PLOT_ERROR_CAP + 20):
        q._record_error(f"job{i}", "boom")

    assert len(q._errors) == server.PLOT_ERROR_CAP
    # FIFO: the oldest went, the newest stayed.
    assert q.error_of("job0") is None
    assert q.error_of(f"job{server.PLOT_ERROR_CAP + 19}") == "boom"


def test_re_recording_an_error_keeps_it_newest():
    """Otherwise a job failing repeatedly would age out while still failing."""
    q = unstarted()
    q._record_error("old", "first")
    for i in range(server.PLOT_ERROR_CAP - 1):
        q._record_error(f"filler{i}", "boom")
    q._record_error("old", "again")
    q._record_error("one-more", "boom")

    assert q.error_of("old") == "again"


def test_a_retry_clears_a_previous_error():
    """So the manifest stops reporting a stale failure the moment the job is
    genuinely pending again."""
    q = unstarted()
    q._errors["a"] = "boom"

    q.enqueue("a")

    assert q.error_of("a") is None


# --------------------------------------------------------------- the drain loop

def test_a_flight_draws_queued_jobs_newest_first():
    async def body():
        flight = PlotFlight(index=0, target=fake_plot_flight.main)
        q = PlotQueue([flight])
        q.enqueue("old")
        q.enqueue("new")
        try:
            await q.start()
            await _drained(q)
        finally:
            await q.aclose()

        assert len(q) == 0
        assert q.error_of("new") is None
        assert q.error_of("old") is None

    asyncio.run(body())


def test_a_failed_draw_is_recorded_and_the_loop_survives():
    """One bad job must not cost the flight the rest of the process's life."""
    async def body():
        flight = PlotFlight(index=0, target=fake_plot_flight.main,
                            target_args=(("fail",),))
        q = PlotQueue([flight])
        q.enqueue("bad")
        try:
            await q.start()
            await _drained(q)
            assert q.error_of("bad") is not None

            # Still alive: a second job goes through.
            q.enqueue("good")
            await _drained(q)
            assert q.error_of("good") is None
        finally:
            await q.aclose()

    asyncio.run(body())


def test_two_flights_both_drain_without_a_lost_wakeup():
    """The shared Event is the one piece of real concurrency here: it wakes every
    waiter, so a flight can clear a set() another flight's work caused. If the
    clear-then-recheck in _drain were the other way round, a batch like this
    would strand jobs with both flights asleep."""
    async def body():
        flights = [PlotFlight(index=i, target=fake_plot_flight.main) for i in range(2)]
        q = PlotQueue(flights)
        try:
            await q.start()
            for i in range(12):
                q.enqueue(f"job{i}")
            await _drained(q)
            assert len(q) == 0
            assert not any(q.error_of(f"job{i}") for i in range(12))
        finally:
            await q.aclose()

    asyncio.run(body())


def test_a_degraded_flight_does_not_spin_the_drain_loop():
    """The failure mode PLOT_RETRY_S exists for. _ensure_usable refuses a
    DEGRADED flight *instantly*, and DEGRADED is terminal until /admin/restart --
    so without a backoff this loop would take the whole queue, fail every job in
    microseconds, and burn a core indefinitely.

    Asserted on the queue rather than on CPU: the jobs must still be *queued*
    after the loop has had time to churn through them, which is only true if the
    loop is sleeping and putting them back.
    """
    async def body():
        flight = PlotFlight(index=0, target=fake_plot_flight.main)
        q = PlotQueue([flight])
        try:
            await q.start()
            await asyncio.wait_for(flight._ready_event.wait(), timeout=30)
            # Force the terminal state the backoff protects against, without
            # waiting out a real crash loop.
            flight._state = server.CoordState.DEGRADED
            flight._degraded_reason = "forced by test"

            for i in range(5):
                q.enqueue(f"job{i}")
            # Several times longer than a spin would need to drain five jobs.
            await asyncio.sleep(0.5)

            # Still queued, not burned through into errors.
            assert len(q) == 5
            assert not any(q.error_of(f"job{i}") for i in range(5))
        finally:
            await q.aclose()

    asyncio.run(body())


def test_a_job_is_eventually_failed_rather_than_retried_forever():
    """A DEGRADED flight is recoverable only via /admin/restart?pool=plot. Until
    someone does that, the honest report is an error the page can show -- not a
    `queued` it polls indefinitely."""
    async def body():
        flight = PlotFlight(index=0, target=fake_plot_flight.main)
        q = PlotQueue([flight])
        # Shorten the backoff so the bounded retries elapse within the test.
        monkey = server.PLOT_RETRY_S
        server.PLOT_RETRY_S = 0.01
        try:
            await q.start()
            await asyncio.wait_for(flight._ready_event.wait(), timeout=30)
            flight._state = server.CoordState.DEGRADED
            flight._degraded_reason = "forced by test"
            q.enqueue("doomed")

            deadline = asyncio.get_running_loop().time() + 15
            while q.error_of("doomed") is None:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("job was retried forever, never failed")
                await asyncio.sleep(0.02)

            assert "gave up after" in q.error_of("doomed")
            assert len(q) == 0
        finally:
            server.PLOT_RETRY_S = monkey
            await q.aclose()

    asyncio.run(body())


def test_an_idle_queue_parks_rather_than_spinning():
    """A busy-wait here would burn a core for every idle plot flight."""
    async def body():
        flight = PlotFlight(index=0, target=fake_plot_flight.main)
        q = PlotQueue([flight])
        try:
            await q.start()
            # Long enough that a spinning loop would have gone round thousands
            # of times; the assertion is that nothing broke and nothing drew.
            await asyncio.sleep(0.3)
            assert q.drawing == {}
            assert len(q) == 0
        finally:
            await q.aclose()

    asyncio.run(body())


def test_a_plot_flight_needs_no_shared_block():
    """The reason PlotFlight subclasses Coord at all: everything but the 513 MB
    block and the re-prime is reusable."""
    async def body():
        flight = PlotFlight(index=0, target=fake_plot_flight.main)
        try:
            await flight.start()
            await asyncio.wait_for(flight._ready_event.wait(), timeout=30)
            assert flight.shm is None
            assert flight.state is server.CoordState.READY
        finally:
            await flight.aclose()

    asyncio.run(body())


async def _drained(q: PlotQueue, timeout: float = 30.0) -> None:
    """Wait until nothing is queued and nothing is drawing."""
    deadline = asyncio.get_running_loop().time() + timeout
    while len(q) or q.drawing:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"queue did not drain: {len(q)} left, {q.drawing}")
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------- the routes

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DONUT_SERVER_TOKEN", TOKEN)
    monkeypatch.setenv("DONUT_SERVER_STAMP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "JOBS", {})
    # A pool with one flight but no drain loop: the routes only ever read
    # len()/state_of()/enqueue(), none of which need a running child.
    monkeypatch.setattr(
        server, "plot_pool", PlotQueue([PlotFlight(index=0, target=fake_plot_flight.main)])
    )
    return TestClient(server.app)


def write_plot(tmp_path, job_id, name, size="full", payload=b"\x89PNG\r\n\x1a\nfake"):
    target = tmp_path / "plots" / job_id
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.{size}.png"
    path.write_bytes(payload)
    return path


def test_the_manifest_reports_absent_for_a_job_with_no_plots(client):
    body = client.get("/result/nope/plots", headers=auth()).json()

    assert body["state"] == "absent"
    assert body["figures"] == {}


def test_the_manifest_reports_ready_and_links_both_sizes(client, tmp_path):
    write_plot(tmp_path, "job1", "donut", "full")
    write_plot(tmp_path, "job1", "donut", "thumb")

    body = client.get("/result/job1/plots", headers=auth()).json()

    assert body["state"] == "ready"
    assert set(body["figures"]["donut"]) == {"full", "thumb"}
    assert body["figures"]["donut"]["thumb"]["url"].endswith("size=thumb")


def test_the_manifest_reports_queued_over_ready(client, tmp_path):
    """A redraw in flight must keep the page polling rather than settling."""
    write_plot(tmp_path, "job1", "donut")
    server.plot_pool.enqueue("job1")

    body = client.get("/result/job1/plots", headers=auth()).json()

    assert body["state"] == "queued"


def test_the_manifest_survives_a_job_the_front_end_never_saw(client, tmp_path):
    """PNGs outlive the process that computed them; JOBS does not."""
    write_plot(tmp_path, "ancient", "donut")

    body = client.get("/result/ancient/plots", headers=auth()).json()

    assert body["state"] == "ready"
    assert body["job_state"] is None


def test_a_failed_redraw_does_not_report_ready(client, tmp_path):
    """The figures on disk are from the *previous* successful draw. Reporting
    `ready` would show the operator six stale PNGs with no hint that the redraw
    they just asked for failed."""
    write_plot(tmp_path, "job1", "donut")
    server.plot_pool._record_error("job1", "boom")

    body = client.get("/result/job1/plots", headers=auth()).json()

    assert body["state"] == "error"
    assert body["error"] == "boom"
    # The stale figures are still reported, so the page can show them *and* say
    # they are stale rather than blanking the section.
    assert "donut" in body["figures"]


def test_pending_outranks_both_ready_and_error(client, tmp_path):
    """A redraw in flight should keep the page polling, not settle on either."""
    write_plot(tmp_path, "job1", "donut")
    server.plot_pool._record_error("job1", "boom")
    server.plot_pool.enqueue("job1")

    assert client.get("/result/job1/plots", headers=auth()).json()["state"] == "queued"


@pytest.mark.parametrize("job_id", ["..", "a.b", "x" * 65])
def test_the_manifest_rejects_a_malformed_job_id(client, job_id):
    """It joins `job_id` onto the stamp dir like the PNG route does, so it gets
    the same guard -- an unchecked one probes for files outside the plots tree."""
    resp = client.get(f"/result/{job_id}/plots", headers=auth())

    assert resp.status_code in (400, 404)


def test_a_plot_is_served_as_a_png(client, tmp_path):
    write_plot(tmp_path, "job1", "wavefront", "thumb", payload=b"thumbbytes")

    resp = client.get("/result/job1/plot/wavefront?size=thumb", headers=auth())

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == b"thumbbytes"


def test_a_missing_plot_is_409_not_404(client):
    """Same contract as /result/{job_id}/images: for the ~17 s after a job is
    DONE this is the ordinary state, and a client should retry."""
    resp = client.get("/result/job1/plot/donut", headers=auth())

    assert resp.status_code == 409


@pytest.mark.parametrize("name", ["bogus", "../../../etc/passwd", "donut.full"])
def test_an_unknown_plot_name_is_rejected(client, name):
    """The whitelist is the only thing standing between a URL segment and the
    filesystem, so it is checked before _plot_path is ever called."""
    resp = client.get(f"/result/job1/plot/{name}", headers=auth())

    assert resp.status_code in (400, 404)


def test_an_unknown_size_is_rejected(client):
    resp = client.get("/result/job1/plot/donut?size=../x", headers=auth())

    assert resp.status_code == 400


@pytest.mark.parametrize("job_id", ["..", "a/b", "a.b", "x" * 65])
def test_a_malformed_job_id_is_rejected(client, job_id):
    resp = client.get(f"/result/{job_id}/plot/donut", headers=auth())

    assert resp.status_code in (400, 404)


def test_enqueueing_one_job_answers_202(client):
    resp = client.post("/admin/plots/job1", headers=auth())

    assert resp.status_code == 202
    assert resp.json()["queued"] is True
    assert server.plot_pool.state_of("job1") == "queued"


def test_enqueueing_a_pending_job_reports_queued_false(client):
    client.post("/admin/plots/job1", headers=auth())

    body = client.post("/admin/plots/job1", headers=auth()).json()

    assert body["queued"] is False
    assert body["queue_depth"] == 1


def test_enqueueing_without_plot_flights_is_409(client, monkeypatch):
    """--num-plot-flights 0 is a legitimate configuration, and a button that
    silently queued into a pool with no drain would be worse than a refusal."""
    monkeypatch.setattr(server, "plot_pool", PlotQueue([]))

    resp = client.post("/admin/plots/job1", headers=auth())

    assert resp.status_code == 409


def test_the_manifest_says_absent_not_queued_without_plot_flights(client, monkeypatch):
    """The symptom the enqueue guard fixes, asserted where it was visible: the
    results page polls on `queued`, so reporting it for a job nothing will draw
    is an infinite poll. `absent` is terminal and offers the (refused) button."""
    monkeypatch.setattr(server, "plot_pool", PlotQueue([]))
    # What the push path does on every completed job.
    server.plot_pool.enqueue("job1")

    body = client.get("/result/job1/plots", headers=auth()).json()

    assert body["state"] == "absent"
    assert body["n_plot_flights"] == 0


def test_backfill_enqueues_only_jobs_missing_plots(client, tmp_path):
    for job in ("done", "todo1", "todo2"):
        (tmp_path / f"{job}.parquet").write_bytes(b"x")
    write_plot(tmp_path, "done", "donut")

    body = client.post("/admin/plots", headers=auth()).json()

    assert body["scanned"] == 3
    assert body["missing"] == 2
    assert body["queued"] == 2
    assert server.plot_pool.state_of("done") is None


def test_backfill_ignores_the_selection_tables(client, tmp_path):
    """They sit in the same directory under the same prefix, and would otherwise
    read as two more jobs apiece."""
    (tmp_path / "job1.parquet").write_bytes(b"x")
    (tmp_path / "job1.detector_images.parquet").write_bytes(b"x")
    (tmp_path / "job1.selection_overlays.parquet").write_bytes(b"x")

    body = client.post("/admin/plots", headers=auth()).json()

    assert body["scanned"] == 1
    assert body["queued"] == 1


def test_backfill_pushes_oldest_first_so_newest_is_drawn_first(client, tmp_path):
    """The push order has to be inverted for a LIFO to pop in the useful order."""
    for i, job in enumerate(("oldest", "middle", "newest")):
        path = tmp_path / f"{job}.parquet"
        path.write_bytes(b"x")
        os.utime(path, (1000 + i, 1000 + i))

    client.post("/admin/plots", headers=auth())

    assert server.plot_pool._take() == "newest"
    assert server.plot_pool._take() == "middle"
    assert server.plot_pool._take() == "oldest"


def test_backfill_without_a_stamp_dir_is_409(client, monkeypatch):
    monkeypatch.delenv("DONUT_SERVER_STAMP_DIR")

    resp = client.post("/admin/plots", headers=auth())

    assert resp.status_code == 409


def test_the_survey_counts_without_enqueueing(client, tmp_path):
    """What the dashboard's backfill note polls. Same counts as the POST, but it
    must leave the queue alone -- it runs on a timer, and a GET that enqueued
    would make opening the page a backfill."""
    for job in ("done", "todo1", "todo2"):
        (tmp_path / f"{job}.parquet").write_bytes(b"x")
    write_plot(tmp_path, "done", "donut")

    body = client.get("/admin/plots", headers=auth()).json()

    assert body["scanned"] == 3
    assert body["missing"] == 2
    assert len(server.plot_pool) == 0


def test_the_survey_works_with_no_plot_flights(client, monkeypatch, tmp_path):
    """Unlike the POST, which has nothing to enqueue into. "14 missing" is what
    an operator running --num-plot-flights 0 wants before deciding to restart."""
    monkeypatch.setattr(server, "plot_pool", PlotQueue([]))
    (tmp_path / "todo1.parquet").write_bytes(b"x")

    resp = client.get("/admin/plots", headers=auth())

    assert resp.status_code == 200
    assert resp.json()["missing"] == 1


def test_the_survey_without_a_stamp_dir_is_409(client, monkeypatch):
    monkeypatch.delenv("DONUT_SERVER_STAMP_DIR")

    resp = client.get("/admin/plots", headers=auth())

    assert resp.status_code == 409


def test_the_survey_and_the_backfill_agree(client, tmp_path):
    """They share one scan so the number the note shows is the number the button
    acts on. A drift between them would have the note disagree with its own
    click."""
    for job in ("done", "todo1", "todo2"):
        (tmp_path / f"{job}.parquet").write_bytes(b"x")
    write_plot(tmp_path, "done", "donut")

    survey = client.get("/admin/plots", headers=auth()).json()
    backfill = client.post("/admin/plots", headers=auth()).json()

    assert (survey["scanned"], survey["missing"]) == (
        backfill["scanned"],
        backfill["missing"],
    )


def test_restart_can_target_the_plot_pool(client):
    """Without this, a DEGRADED plot flight stays down for the life of the
    process: Coord treats DEGRADED as terminal until a restart, and the bare
    route only ever reached pool.flights."""
    resp = client.post("/admin/restart?pool=plot", headers=auth())

    assert resp.status_code == 202
    body = resp.json()
    assert body["pool"] == "plot"
    assert [f["index"] for f in body["flights"]] == [0]


def test_restart_still_defaults_to_the_zernike_pool(client):
    """The pre-existing contract: a bare POST means the compute flights."""
    resp = client.post("/admin/restart", headers=auth())

    assert resp.status_code == 202
    assert resp.json()["pool"] == "zernike"


def test_restart_rejects_an_unknown_pool(client):
    resp = client.post("/admin/restart?pool=bogus", headers=auth())

    assert resp.status_code == 400


def test_restart_validates_the_flight_index_against_the_named_pool(client):
    """The two pools have different lengths, so the bound has to follow `pool`:
    `?pool=plot&flight=1` with one plot flight is out of range even when the
    Zernike pool has two."""
    resp = client.post("/admin/restart?pool=plot&flight=5", headers=auth())

    assert resp.status_code == 400


def test_restarting_an_empty_plot_pool_is_409(client, monkeypatch):
    monkeypatch.setattr(server, "plot_pool", PlotQueue([]))

    resp = client.post("/admin/restart?pool=plot", headers=auth())

    assert resp.status_code == 409


def test_health_reports_the_plot_pool(client):
    server.plot_pool.enqueue("job1")

    body = client.get("/health", headers=auth()).json()

    assert body["plot_queue"]["depth"] == 1
    assert len(body["plot_flights"]) == 1
    # Pool-wide, since one stack feeds every lane -- and emphatically not flight
    # 0's *Zernike* queue depth, which is what a shared _flight_health lookup
    # used to report here.
    assert body["plot_flights"][0]["queue_depth"] == 1


def test_num_plot_flights_zero_means_no_pool():
    """A legitimate configuration, not a degenerate one: the coordinator reads the
    same variable and skips ~35 MB a job of tables nothing would read."""
    import os as _os

    from lsst.ts.donut_server import server as srv

    old = _os.environ.get("DONUT_SERVER_NUM_PLOT_FLIGHTS")
    _os.environ["DONUT_SERVER_NUM_PLOT_FLIGHTS"] = "0"
    try:
        assert srv.num_plot_flights() == 0
    finally:
        if old is None:
            del _os.environ["DONUT_SERVER_NUM_PLOT_FLIGHTS"]
        else:
            _os.environ["DONUT_SERVER_NUM_PLOT_FLIGHTS"] = old


@pytest.mark.parametrize(
    "value, expected", [("0", False), ("1", True), ("3", True), ("junk", True), (None, True)]
)
def test_the_coordinator_agrees_about_whether_plotting_is_on(monkeypatch, value, expected):
    """The two sides must read the same variable the same way, or the coordinator
    skips tables a running plot flight then cannot find (or writes 35 MB a job
    that nothing reads). Imported lazily: this is the one assertion here that
    needs the coordinator module, hence the LSST stack."""
    coordinator = pytest.importorskip("lsst.ts.donut_server.coordinator")
    if value is None:
        monkeypatch.delenv("DONUT_SERVER_NUM_PLOT_FLIGHTS", raising=False)
    else:
        monkeypatch.setenv("DONUT_SERVER_NUM_PLOT_FLIGHTS", value)

    assert coordinator.plotting_enabled() is expected


def test_a_job_is_enqueued_for_plotting_when_its_push_completes(client, monkeypatch):
    """The automatic path. Asserted on the queue rather than through /push, which
    would need a coordinator: the call site is one line in _receive_and_dispatch."""
    record = JobRecord(job_id="job1", state=JobState.DONE)
    monkeypatch.setattr(server, "JOBS", {"job1": record})

    server.plot_pool.enqueue("job1")

    assert server.plot_pool.state_of("job1") == "queued"
