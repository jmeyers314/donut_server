"""Replay metadata substitution: which exposure, and from where.

Separate from `tests/test_trigger.py` because importing `replay` reaches the
stack (it reads FITS headers), which that module's tests deliberately do not --
`trigger` stays importable without `lsst.afw` so the camera's hook pays only
for Python startup, and a test that imported it there would not notice if that
broke.

The header-reading half is not covered here; it needs real files and lives in
`tests/test_daq.py`. What is covered is the decision logic, which plain dicts
exercise completely.
"""
import os

import pytest

from lsst.ts.donut_server import replay

SENSORS = [
    f"R{raft}_SW{half}" for raft in ("00", "04", "40", "44") for half in (0, 1)
]

# Distinct from None: "the card is not there at all" versus "it is there and
# undefined", which the sidecars render differently.
_ABSENT = object()


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
        assert replay.read_emuimage(self.headers()) == self.ORIGINAL

    def test_absent_on_one_sensor_names_it(self):
        """Partial is not good enough: if one sensor cannot say what it is a
        replay of, we cannot be sure the other seven are the same replay."""
        with pytest.raises(RuntimeError) as excinfo:
            replay.read_emuimage(self.headers(R40_SW1=_ABSENT))

        message = str(excinfo.value)
        assert "R40_SW1" in message
        assert "EMUIMAGE" in message

    def test_absent_everywhere_is_real_camera_data(self):
        """Real camera files carry no marker at all, and that is the signal to
        substitute nothing -- not an error, since this is production."""
        assert replay.read_emuimage({s: {} for s in SENSORS}) is None

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_counts_as_absent(self, blank):
        """The plausible shape of a keyword the emulator declared but never
        filled. Treating it as a name would resolve to the replay dir itself."""
        with pytest.raises(RuntimeError, match="absent or blank"):
            replay.read_emuimage(self.headers(R00_SW0=blank))

    def test_none_counts_as_absent(self):
        """How the sidecars render an undefined card, so a header built from
        one can carry None rather than a missing key."""
        with pytest.raises(RuntimeError, match="absent or blank"):
            replay.read_emuimage(self.headers(R00_SW0=None))

    def test_disagreement_carries_both_values(self):
        """Two replays' files in one directory. Silent otherwise: substitution
        would succeed and every downstream cross-check would pass."""
        other = "MC_O_20250530_000999"
        with pytest.raises(RuntimeError) as excinfo:
            replay.read_emuimage(self.headers(R04_SW0=other, R04_SW1=other))

        message = str(excinfo.value)
        assert self.ORIGINAL in message and other in message
        assert "R04_SW0" in message
        assert "disagree" in message

    def test_whitespace_is_stripped(self):
        assert replay.read_emuimage(
            self.headers(f"  {self.ORIGINAL}  ")
        ) == self.ORIGINAL



class TestSubstitutionRouting:
    """Whether to substitute at all, and from which exposure.

    The header *reading* is stubbed out -- that needs real files and is covered
    in `tests/test_daq.py`. What is under test is the decision, which is where
    a replay silently matched to the wrong original would come from.
    """

    @pytest.fixture
    def stubbed(self, monkeypatch, tmp_path):
        """Stubs both header reads and records what was looked up.

        `live` is the header map the live-side stub serves, so a test can
        empty the marker to stand in for real camera data. `loaded` overrides
        the OBSID the substitutes come back carrying, which is how the two can
        be made to disagree.
        """
        asked = {
            "live": {s: {"EMUIMAGE": "FROM_EMUIMAGE"} for s in SENSORS},
            "loaded": None,
        }

        def fake_replay_headers(replay_dir, exposure):
            asked["exposure"] = exposure
            return {s: {"OBSID": asked["loaded"] or exposure} for s in SENSORS}

        monkeypatch.setattr(replay, "read_live_headers", lambda paths: asked["live"])
        monkeypatch.setattr(replay, "replay_headers", fake_replay_headers)
        monkeypatch.setattr(replay.daq, "sw_paths", lambda directory: [])
        monkeypatch.delenv(replay.REPLAY_DIR_VAR, raising=False)
        return asked

    def test_no_marker_substitutes_nothing(self, stubbed, tmp_path):
        """The real camera path: the headers on disk are the truth, so
        resolve_exposure_dir must be handed no substitutes at all."""
        stubbed["live"] = {s: {} for s in SENSORS}

        assert replay.substituted_headers(str(tmp_path)) is None
        assert "exposure" not in stubbed

    def test_no_marker_ignores_the_variable(self, stubbed, tmp_path, monkeypatch):
        """What lets the variable be part of a standing test-stand setup: real
        camera data takes the production path whether or not a tree is
        configured, so the two deployments differ only in their data."""
        stubbed["live"] = {s: {} for s in SENSORS}
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))

        assert replay.substituted_headers(str(tmp_path)) is None
        assert "exposure" not in stubbed

    def test_emuimage_picks_the_exposure(self, stubbed, tmp_path, monkeypatch):
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))

        replay.substituted_headers(str(tmp_path))

        assert stubbed["exposure"] == "FROM_EMUIMAGE"

    def test_marker_with_no_replay_dir_is_refused(self, stubbed, tmp_path):
        """A replay with nowhere to read originals from must not run. The
        alternative is submitting it under BTS's pointing, which comes back
        DONE and wrong -- the whole failure this module exists to prevent."""
        with pytest.raises(RuntimeError) as excinfo:
            replay.substituted_headers(str(tmp_path))

        message = str(excinfo.value)
        assert replay.REPLAY_DIR_VAR in message
        assert "FROM_EMUIMAGE" in message

    def test_replay_dir_that_is_not_a_directory(self, stubbed, monkeypatch, tmp_path):
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path / "nope"))

        with pytest.raises(RuntimeError, match="not a directory"):
            replay.substituted_headers(str(tmp_path))

    def test_the_report_goes_wherever_the_caller_sends_it(
        self, stubbed, tmp_path, monkeypatch
    ):
        """Returned as lines rather than logged, because the trigger prints them
        on stdout and the coordinator logs them -- and they are the only place
        a wrong substitution is visible for review."""
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))
        lines = []

        replay.substituted_headers(str(tmp_path), report=lines.append)

        assert any("names original exposure FROM_EMUIMAGE" in line for line in lines)
        assert any("live -> substituted" in line for line in lines)
        # The marker and the exposure actually loaded agree, so no MISMATCH.
        assert not any("MISMATCH" in line for line in lines)

    def test_a_marker_mismatch_is_called_out(self, stubbed, tmp_path, monkeypatch):
        """The marker named one exposure and the headers loaded say another --
        a replay tree whose directory name disagrees with the OBSID inside it.
        Not refused, since the pixels may still be the right ones, but it must
        not pass unremarked."""
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))
        stubbed["loaded"] = "SOMETHING_ELSE"
        lines = []

        replay.substituted_headers(str(tmp_path), report=lines.append)

        assert any("MISMATCH" in line for line in lines)
