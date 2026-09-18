"""Exposure selection is pure string logic -- no afw, no I/O, no raw files needed."""
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
