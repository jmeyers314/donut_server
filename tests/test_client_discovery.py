"""Exposure selection is pure string logic -- no afw, no I/O, no raw files needed."""
import argparse

import pytest

from lsst.ts.donut_server import client

VISITS = {
    2026060700680: "y",
    2026071300478: "r",
    2026071300535: "i",
}
DETECTORS = [191, 192, 195, 196, 199, 200, 203, 204]


@pytest.fixture
def raw_dir(tmp_path):
    for visit, band in VISITS.items():
        for det in DETECTORS:
            (tmp_path / f"raw_{visit}_{det}_{band}.fits").touch()
    (tmp_path / "notes.txt").touch()  # ignored
    return str(tmp_path)


def test_discover_groups_by_visit(raw_dir):
    found = client.discover_exposures(raw_dir)
    assert set(found) == set(VISITS)
    for visit, band in VISITS.items():
        got_band, paths = found[visit]
        assert got_band == band
        assert sorted(paths) == DETECTORS


def test_discover_empty_dir(tmp_path):
    assert client.discover_exposures(str(tmp_path)) == {}


@pytest.mark.parametrize("visit,band", VISITS.items())
def test_resolve_reports_band_of_visit(raw_dir, visit, band):
    """Band tracks the exposure asked for, rather than anything a caller supplies
    -- it is what /prepare turns into flat_<det>_<band>.fits."""
    got_visit, got_band, paths = client.resolve_exposure(raw_dir, visit)
    assert (got_visit, got_band) == (visit, band)
    assert sorted(paths) == DETECTORS


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
