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

    def test_absent_everywhere_is_the_real_data_case(self):
        """What pointing the replay path at real camera data looks like. The
        message has to say so, since the fix is to unset the variable rather
        than anything about the data."""
        with pytest.raises(RuntimeError) as excinfo:
            replay.read_emuimage({s: {} for s in SENSORS})

        message = str(excinfo.value)
        assert replay.REPLAY_DIR_VAR in message
        assert replay.REPLAY_OBSID_VAR in message

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
    """Which exposure the substitutes are read from: unset / pinned / EMUIMAGE.

    The header *reading* is stubbed out -- that needs real files and is covered
    in `tests/test_daq.py`. What is under test is the decision, which is where
    a replay silently matched to the wrong original would come from.
    """

    @pytest.fixture
    def stubbed(self, monkeypatch, tmp_path):
        """Records which exposure `replay_headers` was asked for."""
        asked = {}

        def fake_replay_headers(replay_dir, exposure):
            asked["exposure"] = exposure
            return {s: {"OBSID": exposure} for s in SENSORS}

        monkeypatch.setattr(
            replay, "read_live_headers",
            lambda paths: {s: {"EMUIMAGE": "FROM_EMUIMAGE"} for s in SENSORS},
        )
        monkeypatch.setattr(replay, "replay_headers", fake_replay_headers)
        monkeypatch.setattr(replay.daq, "sw_paths", lambda directory: [])
        monkeypatch.delenv(replay.REPLAY_OBSID_VAR, raising=False)
        monkeypatch.delenv(replay.REPLAY_DIR_VAR, raising=False)
        return asked

    def test_unset_var_substitutes_nothing(self, stubbed, tmp_path):
        """The real camera path: the headers on disk are the truth, so
        resolve_exposure_dir must be handed no substitutes at all."""
        assert replay.substituted_headers(str(tmp_path)) is None
        assert "exposure" not in stubbed

    def test_emuimage_picks_the_exposure(self, stubbed, tmp_path, monkeypatch):
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))

        replay.substituted_headers(str(tmp_path))

        assert stubbed["exposure"] == "FROM_EMUIMAGE"

    def test_pinned_obsid_beats_emuimage(self, stubbed, tmp_path, monkeypatch):
        """The escape hatch, and the whole reason it exists: a replay whose
        marker is missing or names something absent is still runnable."""
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))
        monkeypatch.setenv(replay.REPLAY_OBSID_VAR, "PINNED_BY_HAND")

        replay.substituted_headers(str(tmp_path))

        assert stubbed["exposure"] == "PINNED_BY_HAND"

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
        """A pinned OBSID that is not what the marker says is legitimate -- it
        is the escape hatch -- but it must not pass unremarked."""
        monkeypatch.setenv(replay.REPLAY_DIR_VAR, str(tmp_path))
        monkeypatch.setenv(replay.REPLAY_OBSID_VAR, "SOMETHING_ELSE")
        lines = []

        replay.substituted_headers(str(tmp_path), report=lines.append)

        assert any("MISMATCH" in line for line in lines)
