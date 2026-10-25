"""The camera trigger's logic that needs no stack and no server: waiting for an
exposure's files to land, reading `EMUIMAGE` to identify which original a BTS
replay is of, and the once-only guard against a hook that fires twice.

Everything here is file-system level on a tmp_path, following
test_client_discovery.py: the parts of trigger.py that import `lsst` are the
submit path, and that is exercised by hand against a running server with
`scripts/replay_drill.py` (see the README), not here.
"""
import os

import pytest

from lsst.ts.donut_server import trigger

SENSORS = ["R00_SW0", "R00_SW1", "R04_SW0", "R04_SW1",
           "R40_SW0", "R40_SW1", "R44_SW0", "R44_SW1"]
OBSID = "MC_O_20260713_000050"

# Distinguishes "this sensor has no EMUIMAGE card at all" from "it has one whose
# value is None", which are different header shapes reaching the same error.
_ABSENT = object()


def write_sensor(directory, sensor, size=1024, sidecar=True, obsid=OBSID):
    """One sensor's FITS, with its sidecar unless told otherwise."""
    fits = directory / f"{obsid}_{sensor}.fits"
    fits.write_bytes(b"\0" * size)
    if sidecar:
        (directory / f"{obsid}_{sensor}.json").write_text("{}")
    return fits


def fill(directory, sensors=SENSORS, **kwargs):
    for sensor in sensors:
        write_sensor(directory, sensor, **kwargs)


class TestPollOnce:
    def test_counts_sensors_with_sidecars(self, tmp_path):
        fill(tmp_path)
        assert sorted(trigger.poll_once(str(tmp_path))) == SENSORS

    def test_a_fits_with_no_sidecar_is_absent(self, tmp_path):
        """Not an error: during the window the camera's script is working on a
        file, that is exactly the state on disk."""
        fill(tmp_path, SENSORS[:7])
        write_sensor(tmp_path, "R44_SW1", sidecar=False)

        found = trigger.poll_once(str(tmp_path))
        assert "R44_SW1" not in found
        assert len(found) == 7

    def test_guider_files_are_ignored(self, tmp_path):
        """The SG sensors carry a *different* FILTER than the SW files of the
        same exposure, so counting them would both overshoot 8 and trip the
        single-filter check in daq.py."""
        fill(tmp_path)
        (tmp_path / f"{OBSID}_R00_SG0_guider.fits").write_bytes(b"\0")
        (tmp_path / f"{OBSID}_R00_SG0_guider.json").write_text("{}")

        assert sorted(trigger.poll_once(str(tmp_path))) == SENSORS

    def test_empty_directory(self, tmp_path):
        assert trigger.poll_once(str(tmp_path)) == {}


class TestWaitForExposure:
    def test_returns_when_complete_and_stable(self, tmp_path):
        fill(tmp_path)
        paths = trigger.wait_for_exposure(
            str(tmp_path), timeout=5, interval=0.01, stable_checks=2
        )
        assert len(paths) == 8
        assert paths == sorted(paths)

    def test_incomplete_times_out_naming_what_is_missing(self, tmp_path):
        """The message is the thing you read when a replay delivers six
        sensors, which is otherwise a very quiet failure."""
        fill(tmp_path, SENSORS[:6])

        with pytest.raises(TimeoutError) as excinfo:
            trigger.wait_for_exposure(str(tmp_path), timeout=0.05, interval=0.01)

        message = str(excinfo.value)
        assert "6/8" in message
        assert "2 missing" in message
        assert "R00_SW0" in message

    def test_empty_directory_times_out(self, tmp_path):
        with pytest.raises(TimeoutError, match="0/8"):
            trigger.wait_for_exposure(str(tmp_path), timeout=0.05, interval=0.01)

    def test_a_growing_file_is_not_stable(self, tmp_path, monkeypatch):
        """Completeness alone cannot start a read: a 9.5 MB FITS that exists is
        not necessarily one that is finished. The last sensor here grows on
        every poll, so it is never stable and the wait must time out rather
        than hand back a half-written file."""
        fill(tmp_path, SENSORS[:7])
        size = [1]

        real = trigger.poll_once

        def growing(directory):
            write_sensor(tmp_path, "R44_SW1", size=size[0])
            size[0] += 1024
            return real(directory)

        monkeypatch.setattr(trigger, "poll_once", growing)

        with pytest.raises(TimeoutError):
            trigger.wait_for_exposure(str(tmp_path), timeout=0.1, interval=0.01)

    def test_a_file_that_settles_becomes_stable(self, tmp_path, monkeypatch):
        """The other half of the above: once writing stops, the wait returns."""
        fill(tmp_path, SENSORS[:7])
        polls = [0]
        real = trigger.poll_once

        def settling(directory):
            polls[0] += 1
            # Grows for the first two polls, then stops.
            write_sensor(tmp_path, "R44_SW1", size=min(polls[0], 3) * 1024)
            return real(directory)

        monkeypatch.setattr(trigger, "poll_once", settling)

        paths = trigger.wait_for_exposure(
            str(tmp_path), timeout=5, interval=0.01, stable_checks=2
        )
        assert len(paths) == 8

    def test_stable_checks_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="stable_checks"):
            trigger.wait_for_exposure(str(tmp_path), stable_checks=0)


def test_importing_trigger_does_not_import_afw():
    """What makes --detach worth having: the parent returns to the camera's
    hook without paying for the stack. `client` reaches `lsst.afw.image`
    through `exposure_codec` (~5 s), so a top-level import of it here would
    silently put that back on the hook's critical path -- measurably, but only
    if somebody is measuring. Hence this test.

    Run in a subprocess because by the time this module is imported, pytest has
    very likely imported afw for some other test.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "from lsst.ts.donut_server import trigger;"
        "bad = [m for m in sys.modules if m.startswith('lsst.afw')];"
        "print(','.join(sorted(bad)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"trigger.py pulled in {out.stdout.strip()}"


class TestCli:
    def test_directory_must_exist(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            trigger.main([str(tmp_path / "nope")])
        assert "not a directory" in capsys.readouterr().err

    def test_remote_host_needs_a_token(self, tmp_path, monkeypatch, capsys):
        """Checked before any polling, so a misconfigured hook fails at once
        rather than after waiting out a 120 s timeout."""
        monkeypatch.delenv("DONUT_SERVER_TOKEN", raising=False)

        with pytest.raises(SystemExit):
            trigger.main([str(tmp_path), "--host", "http://example.invalid:8000"])
        assert "--token is required" in capsys.readouterr().err

    def test_dry_run_needs_a_token_like_anything_else(self, tmp_path, monkeypatch):
        """It used not to -- it resolved locally and contacted nobody. Now the
        *server* resolves, so a dry run is a request like any other and the
        exemption would be a hole rather than a convenience."""
        monkeypatch.delenv("DONUT_SERVER_TOKEN", raising=False)

        with pytest.raises(SystemExit):
            trigger.main([
                str(tmp_path), "--host", "http://example.invalid:8000",
                "--dry-run", "--timeout", "0.05", "--poll-interval", "0.01",
            ])

    @pytest.mark.parametrize(
        "flag,value",
        [("--timeout", "0"), ("--poll-interval", "0"),
         ("--stable-checks", "0"), ("-n", "0")],
    )
    def test_nonsense_numbers_are_rejected(self, tmp_path, flag, value):
        with pytest.raises(SystemExit):
            trigger.main([str(tmp_path), flag, value])

    def test_a_timeout_is_logged_not_tracebacked(self, tmp_path, capsys):
        """When detached, a traceback goes to a log file nobody is reading and
        the exit status goes nowhere, so the message has to carry the cause."""
        assert trigger.main([
            str(tmp_path), "--timeout", "0.05", "--poll-interval", "0.01",
        ]) == 1
        assert "timed out" in capsys.readouterr().out

    def test_detach_runs_the_work_in_a_new_session(self, tmp_path, monkeypatch):
        """The parent must return without importing the stack, so it is the
        child that does everything -- including choosing a log name."""
        calls = {}

        def fake_popen(argv, **kwargs):
            calls["argv"] = argv
            calls["kwargs"] = kwargs
            return None

        monkeypatch.setattr(trigger.subprocess, "Popen", fake_popen)

        logs = tmp_path / "logs"
        assert trigger.main([
            str(tmp_path), "--detach", "--log-dir", str(logs),
        ]) == 0

        assert calls["kwargs"]["start_new_session"] is True
        assert calls["kwargs"]["env"][trigger.DETACHED_VAR] == "1"
        assert (logs / f"{tmp_path.name}.log").exists()

    def test_the_detached_child_does_not_detach_again(self, tmp_path, monkeypatch):
        """The sentinel is what stops an infinite fork chain."""
        monkeypatch.setenv(trigger.DETACHED_VAR, "1")

        def fail(*args, **kwargs):
            raise AssertionError("the child must not re-detach")

        monkeypatch.setattr(trigger.subprocess, "Popen", fail)

        # Reaches the real run and times out on the empty directory.
        assert trigger.main([
            str(tmp_path), "--detach", "--log-dir", str(tmp_path / "logs"),
            "--timeout", "0.05", "--poll-interval", "0.01",
        ]) == 1


class TestNotifier:
    """Poll, then POST a directory and nothing else -- the only path there is.

    Stack-free, like everything else in this module -- which is the property
    under test as much as the behaviour, since a stray import here would undo
    the only thing this path exists for.
    """

    def test_loopback_detection_matches_the_clients(self):
        """Duplicated from `client.host_is_loopback` to avoid importing it, so
        the two must agree -- a divergence would mean one of them is wrong
        about when a token is required."""
        from lsst.ts.donut_server import client

        for host in (
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://example.invalid:8000",
            "http://8.8.8.8:8000",
            "not a url",
        ):
            assert trigger._host_is_loopback(host) == client.host_is_loopback(host), (
                host
            )

    def test_a_remote_host_still_needs_a_token(self, tmp_path, monkeypatch):
        fill(tmp_path)
        monkeypatch.delenv("DONUT_SERVER_TOKEN", raising=False)
        with pytest.raises(SystemExit):
            trigger.main(
                [str(tmp_path), "--host", "http://8.8.8.8:8000"]
            )

    def test_an_unreachable_server_fails_without_a_traceback(
        self, tmp_path, capsys
    ):
        """The hook reads a log, not a stack trace, and a connection refused is
        the most likely failure of all: the server simply is not up."""
        fill(tmp_path)
        # Port 1 is reserved and nothing listens there.
        status = trigger.main(
            [str(tmp_path), "--host", "http://127.0.0.1:1",
             "--timeout", "5"]
        )
        assert status == 1
        assert "cannot reach" in capsys.readouterr().out

    def test_the_notifier_posts_the_directory_and_reports_the_exposure(
        self, tmp_path, monkeypatch, capsys
    ):
        """What the server is told, and what the hook's log records back."""
        import json

        fill(tmp_path)
        sent = {}

        class FakeResponse:
            def read(self):
                return json.dumps({
                    "state": "DONE",
                    "exposure": {
                        "visit": 2026071300150,
                        "physical_filter": "r_57",
                        "imgtype": "acq",
                    },
                }).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            sent["url"] = request.full_url
            sent["body"] = json.loads(request.data)
            sent["auth"] = request.headers.get("Authorization")
            return FakeResponse()

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        status = trigger.main(
            [str(tmp_path), "-n", "16", "--token", "secret"]
        )

        assert status == 0
        assert sent["url"].endswith("/submit?num_workers=16")
        assert sent["body"] == {"directory": str(tmp_path)}
        assert sent["auth"] == "Bearer secret"
        out = capsys.readouterr().out
        assert "visit=2026071300150" in out
        assert "filter=r_57" in out
        assert "DONE" in out
