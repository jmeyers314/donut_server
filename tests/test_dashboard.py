"""Dashboard endpoints and the log-reading helpers.

Imports no part of the LSST stack, like test_coord_lifecycle.

Every test here builds `TestClient(server.app)` *without* the `with` block, on
purpose: entering the context manager runs the lifespan (server.py:668), which
calls start() on the module-global real Coord -- a 513 MB shared block and an
actual coordinator that imports afw. None of these tests need a coordinator, and
none of the routes under test touch one.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from lsst.ts.donut_server import logtail
from lsst.ts.donut_server import server
from lsst.ts.donut_server import table_view
from lsst.ts.donut_server.server import JobRecord, JobState

TOKEN = "test-token"


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def make_client(monkeypatch, jobs=None, peer=None):
    """A client whose peer is *not* loopback unless asked.

    TestClient's default peer is ("testclient", 50000), which is not an IP address at
    all, so server._is_loopback rejects it and every test here exercises the remote
    path -- the one where the token is still required. `peer` opts into the local
    path, which needs no token.
    """
    monkeypatch.setenv("DONUT_SERVER_TOKEN", TOKEN)
    monkeypatch.setattr(server, "JOBS", jobs if jobs is not None else {})
    return TestClient(server.app, client=peer) if peer else TestClient(server.app)


def test_the_dashboard_page_is_localhost_only(monkeypatch):
    """Two failures at once: a wrong DASHBOARD_HTML path fails at request time rather
    than import time, so nothing else would catch a typo or a static/ missing from an
    install; and the page must not be served to a remote browser, which would render a
    dashboard whose every fetch then fails."""
    client = make_client(monkeypatch, peer=("127.0.0.1", 40000))
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # Both state machines must actually be in the page the route serves.
    assert 'id="c-degraded"' in resp.text
    assert 'id="j-COMPUTING"' in resp.text
    # No credential input survives: a masked field here is captured by password
    # managers, which is why the token was removed from this page.
    assert "type=\"password\"" not in resp.text

    # A remote caller is refused outright, with or without a valid token.
    remote = make_client(monkeypatch, peer=("10.0.0.4", 40000))
    assert remote.get("/dashboard").status_code == 403
    assert remote.get("/dashboard", headers=auth()).status_code == 403
    assert remote.get("/results/j1").status_code == 403


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.5", "::1"])
def test_a_loopback_caller_needs_no_token(monkeypatch, host):
    """The point of the change: an operator on the server's own host, or through an
    ssh tunnel (which reconnects from 127.0.0.1), is authorized by being local.

    Parametrized because the loopback block is all of 127/8 plus ::1, and a string
    compare against "127.0.0.1" would pass the first case and fail the others.
    """
    client = make_client(monkeypatch, {"j1": JobRecord(job_id="j1", state=JobState.DONE)},
                         peer=(host, 40000))

    resp = client.get("/admin/jobs")
    assert resp.status_code == 200, resp.text
    assert resp.json()["jobs"][0]["job_id"] == "j1"

    # /health gives a local caller the full detail, not the minimal public body.
    body = client.get("/health").json()
    assert {"n_flights", "n_ready", "flights", "jobs"} <= set(body)
    assert {"pid", "in_flight", "last_loss"} <= set(body["flights"][0])


def test_a_remote_caller_still_needs_the_token(monkeypatch):
    """The regression that matters most: exempting loopback must not open the producer
    API, which is reached over the network.

    The X-Forwarded-For case is the whole security argument. uvicorn only honours that
    header when the immediate peer is already within forwarded_allow_ips (127.0.0.1 by
    default), so a remote caller cannot use it to promote itself to loopback. Asserted
    here because nothing else would notice if a future change trusted the header --
    the result would be an unauthenticated /push open to anyone who can route to it.
    """
    client = make_client(monkeypatch, {"j1": JobRecord(job_id="j1", state=JobState.DONE)},
                         peer=("10.0.0.4", 40000))
    spoof = {"X-Forwarded-For": "127.0.0.1"}

    for path in ("/admin/jobs", "/status/j1", "/result/j1", "/result/j1/table"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers=spoof).status_code == 401, path

    assert client.post("/prepare", json={}).status_code == 401
    assert client.post("/prepare", json={}, headers=spoof).status_code == 401
    assert client.post("/push/j1", content=b"").status_code == 401
    assert client.post("/push/j1", content=b"", headers=spoof).status_code == 401

    # And the token still works from a remote peer -- this is not a lockout.
    assert client.get("/admin/jobs", headers=auth()).status_code == 200


def test_admin_jobs_requires_a_token_and_never_returns_parquet(monkeypatch):
    """Two things at once, both otherwise unguarded.

    No other test in this suite asserts a 401 on any route -- they all send auth()
    -- so this is the only check that a new route was wired up with check_auth at
    all. And a row that serialized table_parquet would be ~200 KB of base64 per
    job, once a second, forever.
    """
    record = JobRecord(job_id="j1", state=JobState.DONE)
    record.table_parquet = b"PAR1xxx"
    record.result = {"summary": {"parquet_bytes": 7}, "quantum": {"visit": 1}}
    client = make_client(monkeypatch, {"j1": record})

    assert client.get("/admin/jobs").status_code == 401

    resp = client.get("/admin/jobs", headers=auth())
    assert resp.status_code == 200
    assert "table_parquet" not in resp.text
    assert "PAR1" not in resp.text

    row = resp.json()["jobs"][0]
    assert row["has_table"] is True
    assert row["parquet_bytes"] == 7
    assert row["table_url"] == "/result/j1/table"


def test_a_wrong_token_401s_the_data_routes_while_health_still_200s(monkeypatch):
    """The asymmetry that made a locked-out dashboard look healthy.

    /health deliberately never 401s: a wrong token just downgrades it to the
    minimal public body (server.py:985). Its status code carries the *coordinator*
    verdict, never an auth verdict -- so it cannot be used as proof of
    authorization, which is what the page's liveness indicator got wrong.
    """
    client = make_client(monkeypatch, {"j1": JobRecord(job_id="j1", state=JobState.DONE)})
    wrong = {"Authorization": "Bearer not-the-token"}

    resp = client.get("/health", headers=wrong)
    assert resp.status_code != 401
    # The detail keys an authed caller gets are all withheld, but the request
    # itself succeeded as far as any client can tell.
    assert set(resp.json()) == {"status", "ready"}

    for path in ("/admin/jobs", "/admin/log", "/admin/log/j1"):
        assert client.get(path, headers=wrong).status_code == 401, path


def test_admin_jobs_is_newest_first_and_capped(monkeypatch):
    """islice(reversed(...)) is one character from wrong, and the wrong direction
    gives a dashboard that silently only ever shows the *oldest* jobs -- which
    looks fine on an empty history and is wrong forever after."""
    jobs = {
        f"job{i}": JobRecord(job_id=f"job{i}", state=JobState.DONE) for i in range(5)
    }
    client = make_client(monkeypatch, jobs)

    body = client.get("/admin/jobs?limit=2", headers=auth()).json()
    assert [row["job_id"] for row in body["jobs"]] == ["job4", "job3"]
    assert body["total"] == 5
    assert body["returned"] == 2

    body = client.get("/admin/jobs?limit=99999", headers=auth()).json()
    assert body["limit"] == server.JOBS_PAGE_MAX
    assert client.get("/admin/jobs?limit=0", headers=auth()).json()["limit"] == 1


def test_admin_jobs_reports_the_job_in_flight(monkeypatch):
    """active_job_ids drives which node the job diagram highlights, and it must skip
    the terminal states rather than just taking the newest rows."""
    jobs = {
        "old": JobRecord(job_id="old", state=JobState.DONE),
        "live": JobRecord(job_id="live", state=JobState.COMPUTING),
        "newest": JobRecord(job_id="newest", state=JobState.ERROR),
    }
    client = make_client(monkeypatch, jobs)
    body = client.get("/admin/jobs", headers=auth()).json()
    assert body["active_job_ids"] == ["live"]
    assert body["counts"] == {"DONE": 1, "COMPUTING": 1, "ERROR": 1}


def test_log_tail_is_bounded_strips_ansi_and_tolerates_a_missing_file(monkeypatch, tmp_path):
    """The seek / partial-line / ANSI-strip path.

    The ANSI case is not hypothetical: LSST tasks colourize their own message text
    and configure_logging does not strip it, so the real log holds SGR sequences
    that would render as literal "[1m[32m" in a <pre>.
    """
    path = tmp_path / "donut.log"
    monkeypatch.setenv("DONUT_SERVER_LOG", str(path))
    client = make_client(monkeypatch)

    # The coordinator configures logging only after its hello, so this is the
    # normal answer for ~15 s on a fresh checkout -- 200, not 500.
    resp = client.get("/admin/log", headers=auth())
    assert resp.status_code == 200
    assert resp.json() == {"path": str(path), "exists": False}

    lines = [f"line {i:05d} " + "x" * 60 for i in range(3000)]
    lines[1000] = "\x1b[1m\x1b[32mcoloured line\x1b[0m"
    path.write_text("\n".join(lines) + "\n")
    assert path.stat().st_size > 200 * 1024

    body = client.get("/admin/log?bytes=8192", headers=auth()).json()
    assert body["exists"] is True
    assert body["truncated"] is True
    assert body["returned_bytes"] <= 8192
    assert "\x1b" not in body["text"]
    # The window is clipped to a whole line at the front, and reaches the end.
    text_lines = body["text"].splitlines()
    assert text_lines[0].startswith("line ")
    assert text_lines[-1] == lines[-1]

    # Below the floor and above the ceiling both clamp rather than being honoured.
    assert client.get("/admin/log?bytes=1", headers=auth()).json()["returned_bytes"] > 1
    huge = client.get("/admin/log?bytes=99999999", headers=auth()).json()
    assert huge["truncated"] is False

    # size is the only cheap change detector on an append-only file.
    size = huge["size"]
    unchanged = client.get(f"/admin/log?since_size={size}", headers=auth()).json()
    assert unchanged == {"unchanged": True, "exists": True, "size": size}


def test_log_slice_cuts_between_markers(monkeypatch, tmp_path):
    """The per-job slice, and its not-found path -- which an operator hits any time
    they click a job older than the search window."""
    path = tmp_path / "donut.log"
    monkeypatch.setenv("DONUT_SERVER_LOG", str(path))
    client = make_client(monkeypatch)

    ids = ["aaaaaaaa-1111", "bbbbbbbb-2222", "cccccccc-3333"]
    out = ["11:00:00 pid=1 INFO coordinator ready, pid 1, logging to x"]
    for jid in ids:
        out.append(f"11:00:01 pid=1 INFO coordinator: job {jid}: starting")
        out.append(f"11:00:02 pid=2 INFO lsst.donutBlitzCorner: filler-for-{jid}")
        out.append(f"11:00:03 pid=2 INFO lsst.donutBlitzCorner: more-for-{jid}")
    path.write_text("\n".join(out) + "\n")

    body = client.get(f"/admin/log/{ids[1]}", headers=auth()).json()
    assert body["found"] is True
    assert f"filler-for-{ids[1]}" in body["text"]
    assert f"filler-for-{ids[0]}" not in body["text"]
    assert f"filler-for-{ids[2]}" not in body["text"]
    # Starts at the marker's own line, not mid-line.
    assert body["text"].splitlines()[0].endswith(f"job {ids[1]}: starting")

    resp = client.get("/admin/log/00000000-0000-0000-0000-000000000000", headers=auth())
    assert resp.status_code == 200
    assert resp.json()["found"] is False

    # A restart also ends a job's output, so the last job's slice stops there.
    path.write_text(
        f"11:00:01 pid=1 INFO coordinator: job {ids[0]}: starting\n"
        f"11:00:02 pid=2 INFO lsst.donutBlitzCorner: filler-for-{ids[0]}\n"
        "11:05:00 pid=9 INFO coordinator: coordinator ready, pid 9, logging to x\n"
        "11:05:01 pid=9 INFO lsst.donutBlitzCorner: after-the-restart\n"
    )
    body = client.get(f"/admin/log/{ids[0]}", headers=auth()).json()
    assert f"filler-for-{ids[0]}" in body["text"]
    assert "after-the-restart" not in body["text"]


def test_log_slice_is_capped(monkeypatch, tmp_path):
    """A crash-looping job emits far more than the measured ~40 KB, and the log is
    never rotated, so the slice needs its own ceiling."""
    path = tmp_path / "donut.log"
    monkeypatch.setenv("DONUT_SERVER_LOG", str(path))
    jid = "dddddddd-4444"
    filler = "\n".join(f"11:00:02 pid=2 INFO noise {i:06d}" for i in range(30000))
    path.write_text(f"11:00:01 pid=1 INFO coordinator: job {jid}: starting\n{filler}\n")
    assert path.stat().st_size > logtail.SLICE_MAX_BYTES

    body = logtail.job_slice(jid)
    assert body["found"] is True
    assert body["truncated"] is True
    assert body["returned_bytes"] == logtail.SLICE_MAX_BYTES


# ------------------------------------------------------------------ results view

# Both hazards are taken from a real donutBlitzCornerResults table: every one of its
# 63 rows contains NaN, and donut_id is ~6.76e18.
BIG_ID = 6761235373898405888


def _results_parquet(n_rows=5):
    """A table shaped like the real one, minus the 33 GB of inputs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    nan = float("nan")
    table = pa.table({
        "det_name": pa.array([f"R00_SW{i % 2}" for i in range(n_rows)]),
        "donut_id": pa.array([BIG_ID + i for i in range(n_rows)], pa.int64()),
        "snr": pa.array([10.0 + i for i in range(n_rows)], pa.float64()),
        # The failed fits: NaN in the scalars and all the way through the vector.
        "fit_dx": pa.array([nan if i % 2 else 0.5 for i in range(n_rows)], pa.float64()),
        "group_fit_success": pa.array([i % 2 == 0 for i in range(n_rows)]),
        "zk_deviation_ccs": pa.array(
            [[nan] * 27 if i % 2 else [0.1 * k for k in range(27)] for i in range(n_rows)],
            pa.list_(pa.float64(), 27)),
    })
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def _done_job(job_id="j1"):
    record = JobRecord(job_id=job_id, state=JobState.DONE)
    record.table_parquet = _results_parquet()
    record.result = {"summary": {"n_rows": 5, "parquet_bytes": len(record.table_parquet)},
                     "quantum": {"visit": 123, "band": "r"}}
    return record


def test_results_rows_survive_nan_and_big_ints(monkeypatch):
    """The two conversions without which this endpoint cannot work at all.

    Starlette's JSONResponse renders with allow_nan=False, so one NaN is a 500 --
    and on a real result *every* row has some, concentrated in the failed fits an
    operator came to look at. Separately, JS parses 6761235373898405888 as
    ...406000, so a bare int64 would hand back a corrupted id.
    """
    client = make_client(monkeypatch, {"j1": _done_job()})
    resp = client.get("/admin/result/j1/rows", headers=auth())
    assert resp.status_code == 200, resp.text
    # The literal tokens json.dumps would emit for a float NaN, which no JSON
    # parser accepts.
    assert "NaN" not in resp.text and "Infinity" not in resp.text

    body = resp.json()
    assert body["stringified"] == ["donut_id"]
    assert body["data"]["donut_id"][0] == str(BIG_ID)
    # Round-tripping through the response must not have lost the low digits.
    assert int(body["data"]["donut_id"][0]) == BIG_ID

    # NaN arrives as null, in scalars and inside vectors alike, and is counted so
    # the UI can say "nan" rather than showing an empty cell.
    assert body["data"]["fit_dx"] == [0.5, None, 0.5, None, 0.5]
    assert body["data"]["zk_deviation_ccs"][1] == [None] * 27
    assert body["nan_counts"]["fit_dx"] == 2
    assert body["nan_counts"]["zk_deviation_ccs"] == 2


def test_results_rows_report_schema_and_page(monkeypatch):
    """The UI lays itself out from the schema, so array width and kind are load
    bearing; and n_rows is unbounded by the protocol, so paging must clamp."""
    client = make_client(monkeypatch, {"j1": _done_job()})
    body = client.get("/admin/result/j1/rows", headers=auth()).json()

    kinds = {c["name"]: c["kind"] for c in body["schema"]}
    assert kinds == {"det_name": "string", "donut_id": "int", "snr": "float",
                     "fit_dx": "float", "group_fit_success": "bool",
                     "zk_deviation_ccs": "array"}
    zk = next(c for c in body["schema"] if c["name"] == "zk_deviation_ccs")
    assert zk["width"] == 27

    page = client.get("/admin/result/j1/rows?offset=2&limit=2", headers=auth()).json()
    assert (page["offset"], page["returned"], page["n_rows"]) == (2, 2, 5)
    assert page["data"]["det_name"] == ["R00_SW0", "R00_SW1"]
    assert client.get("/admin/result/j1/rows?limit=99999",
                      headers=auth()).json()["limit"] == table_view.PAGE_MAX

    # A column subset keeps the schema whole but trims the payload, so opening one
    # 67-wide Zernike column does not mean shipping all thirteen.
    thin = client.get("/admin/result/j1/rows?columns=det_name,snr", headers=auth()).json()
    assert set(thin["data"]) == {"det_name", "snr"}
    assert len(thin["schema"]) == 6


def test_results_rows_and_page_reject_the_unviewable(monkeypatch):
    """Every state that has no table to show, plus auth."""
    jobs = {
        "done": _done_job("done"),
        "computing": JobRecord(job_id="computing", state=JobState.COMPUTING),
        "failed": JobRecord(job_id="failed", state=JobState.ERROR, error="boom"),
    }
    client = make_client(monkeypatch, jobs)

    assert client.get("/admin/result/done/rows").status_code == 401
    assert client.get("/admin/result/nope/rows", headers=auth()).status_code == 404
    # 409 rather than 404: the job is real, it just has nothing to view yet.
    assert client.get("/admin/result/computing/rows", headers=auth()).status_code == 409
    failed = client.get("/admin/result/failed/rows", headers=auth())
    assert failed.status_code == 409
    assert "boom" in failed.text

    # The page itself is the credential-free shell, like /dashboard: served on
    # loopback, refused elsewhere (which this client's peer is).
    local = make_client(monkeypatch, jobs, peer=("127.0.0.1", 40000))
    page = local.get("/results/done")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "/admin/result/" in page.text


# ------------------------------------------------------- config overrides (-c/-C)

# A -C body long enough that finding it echoed anywhere would be unambiguous.
SECRET_BODY = "config.maxFitScatter = 2.0  # UNIQUE_OVERRIDE_BODY_MARKER\n"

DUMP = "import lsst.ts.wep.blitz.donutBlitzCorner\nconfig.maxFitScatter = 2.0\n"


class FakeCoord:
    """Just enough Coord surface for the config routes: no child, no shared block."""

    def __init__(self, snapshot=None, generation=1, primed_args=None, state=None):
        self._snapshot = snapshot
        self.generation = generation
        self.primed_args = primed_args
        self.state = state or server.CoordState.READY
        self.index = 0

    @property
    def config_snapshot(self):
        return self._snapshot

    # /health reads these too.
    def is_alive(self):
        return True

    spawns = 1
    restart_attempts = 0
    child_pid = 4242
    child_uptime_s = 12.5
    last_loss = None
    degraded_reason = None
    in_flight = 0
    busy = False


def patch_pool(monkeypatch, *coords):
    """Swap in a FlightPool over the given fakes, bypassing its own construction
    (which would spawn real coordinators)."""
    pool = server.FlightPool.__new__(server.FlightPool)
    pool.flights = list(coords)
    for i, coord in enumerate(coords):
        coord.index = i
    pool._by_job = {}
    pool._last_used = [0.0] * len(coords)
    pool._queued = [0] * len(coords)
    monkeypatch.setattr(server, "pool", pool)
    return pool


def make_snapshot(generation=1, overrides=None):
    import hashlib

    return server.ConfigSnapshot(
        dump=DUMP,
        generation=generation,
        overrides=overrides if overrides is not None else [],
        at=1_700_000_000.0,
        sha256=hashlib.sha256(DUMP.encode()).hexdigest(),
    )


def test_config_needs_a_token_and_409s_before_any_prepare(monkeypatch):
    client = make_client(monkeypatch)
    patch_pool(monkeypatch, FakeCoord(snapshot=None, state=server.CoordState.STARTING))

    assert client.get("/config").status_code == 401

    # 409 rather than 404 or 503: the route is real and the coordinator may be
    # healthy -- it simply has not prepared yet. The state is named so an operator
    # can tell "still booting" from "up but never primed".
    resp = client.get("/config", headers=auth())
    assert resp.status_code == 409
    assert "starting" in resp.text


def test_config_serves_a_loadable_file_with_a_freshness_verdict(monkeypatch):
    client = make_client(monkeypatch)
    snap = make_snapshot(generation=3)
    patch_pool(monkeypatch, FakeCoord(snap, generation=3, primed_args={"cmd": "prepare"}))

    resp = client.get("/config", headers=auth())

    assert resp.status_code == 200
    # text/plain and a .py filename, so a saved copy round-trips through
    # Config.load -- which is why the staleness verdict is in headers, not the body.
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.headers["content-disposition"].endswith('filename="donutBlitzCornerConfig.py"')
    assert resp.text == DUMP
    assert resp.headers["x-donut-stale"] == "0"
    assert resp.headers["x-donut-generation"] == "3"
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "coord_kwargs",
    [
        # The snapshot predates the live child: a restart whose re-prime never landed.
        {"generation": 4, "primed_args": {"cmd": "prepare"}},
        # Or the coordinator is up but no longer primed at all.
        {"generation": 3, "primed_args": None},
    ],
)
def test_config_flags_a_snapshot_that_may_not_describe_the_live_child(monkeypatch, coord_kwargs):
    client = make_client(monkeypatch)
    patch_pool(monkeypatch, FakeCoord(make_snapshot(generation=3), **coord_kwargs))

    resp = client.get("/config", headers=auth())

    # Still 200 with the bytes: a stale answer is far more useful than none, as long
    # as it says so.
    assert resp.status_code == 200
    assert resp.headers["x-donut-stale"] == "1"


def test_config_is_still_served_while_degraded(monkeypatch):
    """The cache exists precisely for this: DEGRADED is when an operator most wants
    to know what the dead coordinator was running, and a command round-trip would
    have nothing to talk to."""
    client = make_client(monkeypatch)
    patch_pool(
        monkeypatch,
        FakeCoord(
            make_snapshot(generation=2),
            generation=2,
            primed_args={"cmd": "prepare"},
            state=server.CoordState.DEGRADED,
        ),
    )

    resp = client.get("/config", headers=auth())

    assert resp.status_code == 200
    assert resp.text == DUMP


def test_health_fingerprints_the_config_without_echoing_a_C_body(monkeypatch):
    """/health is polled at 1 Hz, so a -C body echoed here would be a permanent
    bandwidth cost -- and the dump itself is ~77 KB, far too big to inline."""
    client = make_client(monkeypatch)
    primed = {
        "cmd": "prepare",
        "band": "r",
        "boresight_ra": 283.666,
        "boresight_dec": -28.1326,
        "config_overrides": [
            {"kind": "value", "field": "maxFitScatter", "value": "2.0"},
            {"kind": "python", "name": "/home/op/tweaks.py", "text": SECRET_BODY},
        ],
    }
    snap = make_snapshot(generation=1, overrides=server._digest_overrides(primed["config_overrides"]))
    patch_pool(monkeypatch, FakeCoord(snap, generation=1, primed_args=primed))

    resp = client.get("/health", headers=auth())
    body = resp.json()

    assert "UNIQUE_OVERRIDE_BODY_MARKER" not in resp.text
    (flight,) = body["flights"]
    assert flight["config"] == {
        "generation": 1,
        "stale": False,
        "bytes": len(DUMP),
        "sha256": snap.sha256[:12],
        "n_overrides": 2,
        "at": 1_700_000_000.0,
    }

    shown = flight["primed_args"]["config_overrides"]
    # A -c entry survives whole: it is what an operator reads off the dashboard.
    assert shown[0] == {"kind": "value", "field": "maxFitScatter", "value": "2.0"}
    # A -C entry keeps its identity and a fingerprint, but not its body.
    assert shown[1]["name"] == "/home/op/tweaks.py"
    assert shown[1]["lines"] == 1
    assert "text" not in shown[1]
    # The fields the dashboard already renders are untouched by the projection.
    assert flight["primed_args"]["band"] == "r"
    assert flight["primed_args"]["boresight_ra"] == 283.666


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ("not-a-list", "must be a list"),
        ([{"field": "x", "value": "1"}], "kind must be one of"),
        ([{"kind": "nope", "field": "x", "value": "1"}], "kind must be one of"),
        # A number here would silently take applyTo's YAML branch instead of its
        # command-line branch, changing -c semantics, so it is refused outright.
        ([{"kind": "value", "field": "maxFitScatter", "value": 2.0}], "must be a string"),
        ([{"kind": "value", "field": "", "value": "1"}], "non-empty string"),
        ([{"kind": "python", "name": "f.py"}], "text must be a string"),
        ([{"kind": "value", "field": "x", "value": "1"}] * 65, "too many"),
        ([{"kind": "python", "text": "#" * (256 * 1024 + 1)}], "over the"),
    ],
)
def test_prepare_rejects_a_malformed_override_list_without_a_coordinator(
    monkeypatch, overrides, expected
):
    """Shape validation happens before send_command, so none of these can reach the
    coordinator -- which is what lets these tests run with no child at all."""
    client = make_client(monkeypatch)

    resp = client.post(
        "/prepare",
        json={"band": "r", "boresight_ra": 0.0, "boresight_dec": 0.0,
              "config_overrides": overrides},
        headers=auth(),
    )

    assert resp.status_code == 400
    assert expected in resp.text


def test_absent_and_empty_override_lists_are_the_same_thing():
    # A bare /prepare is the documented reset for a bad override set, so "omitted"
    # must not mean something different from "explicitly empty".
    assert server._parse_overrides({}) == []
    assert server._parse_overrides({"config_overrides": None}) == []
    assert server._parse_overrides({"config_overrides": []}) == []
