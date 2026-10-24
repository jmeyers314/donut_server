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


class TestReadEmuimage:
    """Which original exposure a replay is of, from the live headers.

    Plain dicts stand in for headers: `read_emuimage` only subscripts them, and
    keeping these stack-free is what makes them run in the fast suite.
    """

    ORIGINAL = "MC_O_20250530_000395"

    def headers(self, value=ORIGINAL, **overrides):
        """All 8 sensors carrying `value`, with per-sensor overrides."""
        out = {s: {"EMUIMAGE": value} for s in SENSORS}
        for sensor, override in overrides.items():
            if override is _ABSENT:
                out[sensor] = {}
            else:
                out[sensor] = {"EMUIMAGE": override}
        return out

    def test_all_eight_agreeing(self):
        assert trigger.read_emuimage(self.headers()) == self.ORIGINAL

    def test_absent_on_one_sensor_names_it(self):
        """Partial is not good enough: if one sensor cannot say what it is a
        replay of, we cannot be sure the other seven are the same replay."""
        with pytest.raises(RuntimeError) as excinfo:
            trigger.read_emuimage(self.headers(R40_SW1=_ABSENT))

        message = str(excinfo.value)
        assert "R40_SW1" in message
        assert "EMUIMAGE" in message

    def test_absent_everywhere_is_the_real_data_case(self):
        """What pointing the replay path at real camera data looks like. The
        message has to say so, since the fix is to unset the variable rather
        than anything about the data."""
        with pytest.raises(RuntimeError) as excinfo:
            trigger.read_emuimage({s: {} for s in SENSORS})

        message = str(excinfo.value)
        assert trigger.REPLAY_DIR_VAR in message
        assert trigger.REPLAY_OBSID_VAR in message

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_counts_as_absent(self, blank):
        """The plausible shape of a keyword the emulator declared but never
        filled. Treating it as a name would resolve to the replay dir itself."""
        with pytest.raises(RuntimeError, match="absent or blank"):
            trigger.read_emuimage(self.headers(R00_SW0=blank))

    def test_none_counts_as_absent(self):
        """How the sidecars render an undefined card, so a header built from
        one can carry None rather than a missing key."""
        with pytest.raises(RuntimeError, match="absent or blank"):
            trigger.read_emuimage(self.headers(R00_SW0=None))

    def test_disagreement_carries_both_values(self):
        """Two replays' files in one directory. Silent otherwise: substitution
        would succeed and every downstream cross-check would pass."""
        other = "MC_O_20250530_000999"
        with pytest.raises(RuntimeError) as excinfo:
            trigger.read_emuimage(self.headers(R04_SW0=other, R04_SW1=other))

        message = str(excinfo.value)
        assert self.ORIGINAL in message and other in message
        assert "R04_SW0" in message
        assert "disagree" in message

    def test_whitespace_is_stripped(self):
        assert trigger.read_emuimage(
            self.headers(f"  {self.ORIGINAL}  ")
        ) == self.ORIGINAL


class TestResolveSourceRouting:
    """Which exposure name `resolve_source` decides to substitute from.

    `daq.resolve_exposure_dir` and `replay_headers` are stubbed out: what is
    under test is the routing between unset / pinned / EMUIMAGE, not the header
    reading, which needs the stack.
    """

    @pytest.fixture
    def stubbed(self, monkeypatch, tmp_path):
        """Records which exposure replay_headers was asked for."""
        asked = {}

        def fake_read_live(paths):
            return {s: {"EMUIMAGE": "FROM_EMUIMAGE"} for s in SENSORS}

        def fake_replay_headers(replay_dir, exposure):
            asked["exposure"] = exposure
            return {s: {"OBSID": exposure} for s in SENSORS}

        import sys
        import types

        # resolve_source imports daq locally; a stub module satisfies it
        # without pulling in the stack.
        fake_daq = types.ModuleType("lsst.ts.donut_server.daq")
        fake_daq.resolve_exposure_dir = lambda directory, headers=None: (
            headers, "stub"
        )
        monkeypatch.setitem(
            sys.modules, "lsst.ts.donut_server.daq", fake_daq
        )
        monkeypatch.setattr(trigger, "read_live_headers", fake_read_live)
        monkeypatch.setattr(trigger, "replay_headers", fake_replay_headers)
        monkeypatch.setattr(trigger, "report_substitution", lambda *a: None)
        monkeypatch.delenv(trigger.REPLAY_OBSID_VAR, raising=False)
        monkeypatch.delenv(trigger.REPLAY_DIR_VAR, raising=False)
        return asked

    def test_unset_var_substitutes_nothing(self, stubbed, tmp_path):
        """The real camera path: headers come from the polled directory, so
        resolve_exposure_dir is called with no substitutes at all."""
        headers, _ = trigger.resolve_source(str(tmp_path), [])

        assert headers is None
        assert "exposure" not in stubbed

    def test_emuimage_picks_the_exposure(self, stubbed, tmp_path, monkeypatch):
        monkeypatch.setenv(trigger.REPLAY_DIR_VAR, str(tmp_path))

        trigger.resolve_source(str(tmp_path), [])

        assert stubbed["exposure"] == "FROM_EMUIMAGE"

    def test_pinned_obsid_beats_emuimage(self, stubbed, tmp_path, monkeypatch):
        """The escape hatch, and the whole reason it exists: a replay whose
        marker is missing or names something absent is still runnable."""
        monkeypatch.setenv(trigger.REPLAY_DIR_VAR, str(tmp_path))
        monkeypatch.setenv(trigger.REPLAY_OBSID_VAR, "PINNED_BY_HAND")

        trigger.resolve_source(str(tmp_path), [])

        assert stubbed["exposure"] == "PINNED_BY_HAND"

    def test_replay_dir_that_is_not_a_directory(self, stubbed, monkeypatch, tmp_path):
        monkeypatch.setenv(trigger.REPLAY_DIR_VAR, str(tmp_path / "nope"))

        with pytest.raises(RuntimeError, match="not a directory"):
            trigger.resolve_source(str(tmp_path), [])


class TestClaim:
    def test_first_claim_wins_and_the_second_does_not(self, tmp_path):
        """A hook that fires twice for one exposure must submit once: a second
        push burns a flight and puts a duplicate exposure on the dashboard for
        the same pixels."""
        state = str(tmp_path / "state")

        assert trigger.claim(state, "2026071300050") is True
        assert trigger.claim(state, "2026071300050") is False

    def test_different_exposures_do_not_collide(self, tmp_path):
        state = str(tmp_path / "state")

        assert trigger.claim(state, "2026071300050") is True
        assert trigger.claim(state, "2026071300150") is True

    def test_marker_records_when_and_by_whom(self, tmp_path):
        state = str(tmp_path / "state")
        trigger.claim(state, "2026071300050")

        text = (tmp_path / "state" / "2026071300050.done").read_text()
        assert f"pid={os.getpid()}" in text


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

    def test_dry_run_needs_no_token(self, tmp_path, monkeypatch):
        """--dry-run contacts no server, so demanding credentials would make
        the first thing you run on a new deployment the hardest."""
        monkeypatch.delenv("DONUT_SERVER_TOKEN", raising=False)

        # Times out on an empty directory rather than erroring on the token.
        assert trigger.main([
            str(tmp_path), "--host", "http://example.invalid:8000",
            "--dry-run", "--timeout", "0.05", "--poll-interval", "0.01",
        ]) == 1

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
