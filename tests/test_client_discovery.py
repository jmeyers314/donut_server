"""Client-side logic that needs no afw, no I/O and no raw files: exposure
selection, -c/-C override parsing, and how a server's error body is surfaced."""
import argparse
import json

import pytest
import requests

from lsst.ts.donut_server import client

VISITS = {
    2026060700680: "y_10",
    2026071300478: "r_57",
    2026071300535: "i_39",
}
DETECTORS = [191, 192, 195, 196, 199, 200, 203, 204]


@pytest.fixture
def raw_dir(tmp_path):
    for visit, physical_filter in VISITS.items():
        for det in DETECTORS:
            (tmp_path / f"raw_{visit}_{det}_{physical_filter}.fits").touch()
    (tmp_path / "notes.txt").touch()  # ignored
    return str(tmp_path)


def test_discover_groups_by_visit(raw_dir):
    found = client.discover_exposures(raw_dir)
    assert set(found) == set(VISITS)
    for visit, physical_filter in VISITS.items():
        got_filter, paths = found[visit]
        assert got_filter == physical_filter
        assert sorted(paths) == DETECTORS


def test_discover_empty_dir(tmp_path):
    assert client.discover_exposures(str(tmp_path)) == {}


@pytest.mark.parametrize("visit,physical_filter", VISITS.items())
def test_resolve_reports_physical_filter_of_visit(raw_dir, visit, physical_filter):
    """The filter tracks the exposure asked for, rather than anything a caller
    supplies -- it is what /prepare turns into
    flat_<det>_<physical_filter>.fits."""
    got_visit, got_filter, paths = client.resolve_exposure(raw_dir, visit)
    assert (got_visit, got_filter) == (visit, physical_filter)
    assert sorted(paths) == DETECTORS


def test_a_raw_the_pattern_cannot_parse_is_an_error(raw_dir, tmp_path):
    """A band-suffixed name is what the pre-physical_filter raws looked like, so
    a half-renamed directory lands here. Skipping it silently would instead yield
    a short detector list or a missing visit much later."""
    (tmp_path / "raw_2026071300478_191_r.fits").touch()

    with pytest.raises(RuntimeError, match="does not parse"):
        client.discover_exposures(raw_dir)


def test_a_filter_with_no_calibs_is_rejected_at_discovery(tmp_path):
    """Caught here rather than as a missing flat_<det>_<filter>.fits deep in the
    coordinator's calib load."""
    for det in DETECTORS:
        (tmp_path / f"raw_2026071300478_{det}_r_75.fits").touch()

    with pytest.raises(RuntimeError, match="no calibs for"):
        client.discover_exposures(str(tmp_path))


def test_resolve_unknown_visit(raw_dir):
    with pytest.raises(RuntimeError, match="visit 999 not found"):
        client.resolve_exposure(raw_dir, 999)


def test_resolve_no_raws(tmp_path):
    with pytest.raises(RuntimeError, match="no raw_\\*.fits files found"):
        client.resolve_exposure(str(tmp_path), 2026071300478)


# --------------------------------------------------------- -c/-C override parsing


@pytest.fixture
def parser():
    """Just the two override options, wired exactly as main() wires them.

    A miniature parser rather than main()'s, which would demand --visit and a raw
    directory; the shared `dest` and the action are the entire mechanism under test.
    """
    p = argparse.ArgumentParser(exit_on_error=False)
    for flag, short in (("--config", "-c"), ("--config-file", "-C")):
        p.add_argument(
            short, flag, dest="config_overrides", action=client.ConfigOverrideAction
        )
    return p


def test_overrides_keep_their_command_line_order(parser, tmp_path):
    """-c and -C interleave, and the last write to a field wins -- so the relative
    order of the two kinds is load-bearing and has to survive into the wire format."""
    path = tmp_path / "tweaks.py"
    path.write_text("config.savePlots = True\n")

    ns = parser.parse_args(["-c", "a=1", "-C", str(path), "-c", "b=2"])

    assert [e["kind"] for e in ns.config_overrides] == ["value", "python", "value"]
    assert ns.config_overrides[0] == {"kind": "value", "field": "a", "value": "1"}
    assert ns.config_overrides[2] == {"kind": "value", "field": "b", "value": "2"}
    # Contents, not the path -- the server may be on another host. The absolute path
    # rides along only so a traceback inside the override names the operator's file.
    assert ns.config_overrides[1]["text"] == "config.savePlots = True\n"
    assert ns.config_overrides[1]["name"] == str(path)


def test_a_value_splits_on_the_first_equals_only(parser):
    ns = parser.parse_args(["-c", "donutSelector.xCoordField=a=b"])

    assert ns.config_overrides == [
        {"kind": "value", "field": "donutSelector.xCoordField", "value": "a=b"}
    ]


def test_no_overrides_defaults_to_none(parser):
    # run_once normalizes this; what matters is that argparse does not share one
    # mutable list across invocations.
    assert parser.parse_args([]).config_overrides is None


@pytest.mark.parametrize("argv", [["-c", "noequals"], ["-c", "=1"]])
def test_a_malformed_value_fails_during_parsing(parser, argv):
    # parser.error exits, which is the point: this lands before ~900 MB of raws are
    # read, and before the --token check.
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_an_unreadable_override_file_fails_during_parsing(parser, tmp_path):
    with pytest.raises(SystemExit):
        parser.parse_args(["-C", str(tmp_path / "nope.py")])


def test_an_oversized_override_file_fails_during_parsing(parser, tmp_path):
    path = tmp_path / "huge.py"
    path.write_text("#" * (client.MAX_OVERRIDE_TEXT_BYTES + 1))

    with pytest.raises(SystemExit):
        parser.parse_args(["-C", str(path)])


# ------------------------------------------------------- error-body surfacing


def make_response(status, body=b"", content_type="application/json"):
    """A `requests.Response` built by hand.

    Not FastAPI's TestClient: that one is httpx-backed, so its
    `raise_for_status` raises `httpx.HTTPStatusError` and would exercise a path
    the real client -- which uses `requests` -- never takes.
    """
    resp = requests.Response()
    resp.status_code = status
    resp.reason = "Internal Server Error"
    resp.url = "http://127.0.0.1:8000/prepare"
    resp._content = body if isinstance(body, bytes) else body.encode()
    resp.headers["Content-Type"] = content_type
    return resp


def test_a_server_error_carries_the_servers_own_explanation():
    """FastAPI's `detail` has to reach the message. Without it a misconfigured
    data directory is a 500 and a traceback with no cause, recoverable only via
    /admin/jobs -- which is the body the client was already holding."""
    reason = "DONUT_SERVER_STAMP_DIR is not set; point it at a writable directory"
    resp = make_response(500, json.dumps({"detail": reason}))

    with pytest.raises(requests.HTTPError) as excinfo:
        client.raise_for_status(resp)

    assert reason in str(excinfo.value)
    # Status and URL survive: this augments requests' message, not replaces it.
    assert "500" in str(excinfo.value)
    # The transient path reads `.response` off the exception, so it must persist.
    assert excinfo.value.response is resp


@pytest.mark.parametrize("key", ["detail", "error", "reason"])
def test_any_of_the_three_body_shapes_is_found(key):
    """`detail` is FastAPI's, but /health-style bodies say `reason` and a job
    record says `error`; all three appear on real responses from this server."""
    resp = make_response(500, json.dumps({key: "the cause"}))

    with pytest.raises(requests.HTTPError, match="the cause"):
        client.raise_for_status(resp)


def test_a_non_json_body_is_surfaced_but_truncated():
    """A proxy's HTML error page, or a crashed worker's plain text, still says
    more than the status line -- but may be arbitrarily long."""
    resp = make_response(502, "x" * 5000, content_type="text/html")

    with pytest.raises(requests.HTTPError) as excinfo:
        client.raise_for_status(resp)

    assert len(str(excinfo.value)) < 1000


def test_an_empty_body_leaves_the_message_alone():
    """Nothing to add must mean no change, not a dangling separator."""
    resp = make_response(500, b"")

    with pytest.raises(requests.HTTPError) as excinfo:
        client.raise_for_status(resp)

    assert str(excinfo.value).endswith("http://127.0.0.1:8000/prepare")


def test_a_success_is_a_no_op():
    assert client.raise_for_status(make_response(200, json.dumps({"job_id": "x"}))) is None


def test_the_transient_path_still_works_off_the_augmented_error(capsys):
    """Loop mode must keep swallowing a 503: the augmented exception carries the
    same `.response`, which is where `reason` and Retry-After are read from."""
    resp = make_response(503, json.dumps({"reason": "coordinator restarting"}))
    resp.headers["Retry-After"] = "5"

    try:
        client.raise_for_status(resp)
    except requests.HTTPError as exc:
        client.report_transient_or_raise(exc)

    out = capsys.readouterr().out
    assert "coordinator restarting" in out and "retry_after=5s" in out
