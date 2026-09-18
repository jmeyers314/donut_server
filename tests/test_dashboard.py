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

from fastapi.testclient import TestClient

from lsst.ts.donut_server import logtail
from lsst.ts.donut_server import server
from lsst.ts.donut_server import table_view
from lsst.ts.donut_server.server import JobRecord, JobState

TOKEN = "test-token"


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def make_client(monkeypatch, jobs=None):
    monkeypatch.setenv("DONUT_SERVER_TOKEN", TOKEN)
    monkeypatch.setattr(server, "JOBS", jobs if jobs is not None else {})
    return TestClient(server.app)


def test_dashboard_page_is_served_without_a_token(monkeypatch):
    """A wrong DASHBOARD_HTML path fails at request time, not import time, so
    nothing else would catch a typo or a static/ missing from an install."""
    client = make_client(monkeypatch)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # Both state machines must actually be in the page the route serves.
    assert 'id="c-degraded"' in resp.text
    assert 'id="j-COMPUTING"' in resp.text


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
    minimal public body (server.py:916). Its status code carries the *coordinator*
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
    """active_job_id drives which node the job diagram highlights, and it must skip
    the terminal states rather than just taking the newest row."""
    jobs = {
        "old": JobRecord(job_id="old", state=JobState.DONE),
        "live": JobRecord(job_id="live", state=JobState.COMPUTING),
        "newest": JobRecord(job_id="newest", state=JobState.ERROR),
    }
    client = make_client(monkeypatch, jobs)
    body = client.get("/admin/jobs", headers=auth()).json()
    assert body["active_job_id"] == "live"
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

    # The page itself is the unauthenticated shell, like /dashboard.
    page = client.get("/results/done")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "/admin/result/" in page.text
