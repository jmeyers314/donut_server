"""DAQ-native amplifier assembly, against an independently-produced reference.

The point of this module is the reference comparison. `daq.load_raw` claims to
reproduce what `LsstCamRawFormatter.readFull()` does for a whole-CCD read
(daq.py's docstring), and both trees needed to check that claim are already on
disk: `$DONUT_SERVER_REPLAY_DIR` holds the DAQ-native amplifier files and
`$DONUT_SERVER_RAW_DIR` holds the same exposures already assembled. Until this
module existed nothing compared them, so the whole assembly path -- and the
substituted-header seam running through it -- could change silently.

Skipped unless both trees are present and share an exposure, since neither is in
version control. `setup.sh` exports DONUT_SERVER_REPLAY_DIR but *not*
DONUT_SERVER_RAW_DIR; see README's "Verification" section.
"""
import glob
import os
import re

import numpy as np
import pytest

from lsst.ts.donut_server import daq

REPLAY_DIR = os.environ.get("DONUT_SERVER_REPLAY_DIR", "")
RAW_DIR = os.environ.get("DONUT_SERVER_RAW_DIR", "")

_RAW_RE = re.compile(r"raw_(\d+)_(\d+)_(.+)\.fits")


def _reference_index() -> dict[tuple[int, int], str]:
    """{(visit, detector): path} for every assembled raw on disk."""
    index = {}
    for path in glob.glob(os.path.join(RAW_DIR, "raw_*.fits")) if RAW_DIR else []:
        m = _RAW_RE.fullmatch(os.path.basename(path))
        if m:
            index[(int(m.group(1)), int(m.group(2)))] = path
    return index


def _find_paired_exposure() -> tuple[str, dict[tuple[int, int], str]] | None:
    """A replay directory whose exposure is also present assembled, or None.

    Matched on the *exposure id read from the headers* rather than on the
    directory name, so this keeps working if the replay tree is re-laid-out.
    """
    index = _reference_index()
    if not index or not REPLAY_DIR:
        return None
    for directory in sorted(glob.glob(os.path.join(REPLAY_DIR, "*"))):
        if not os.path.isdir(directory):
            continue
        try:
            source, _ = daq.resolve_exposure_dir(directory)
        except Exception:
            # A directory that does not resolve is not this module's business;
            # it is only a candidate for pairing.
            continue
        if all((source.visit, det) in index for det in source.handles):
            return directory, index
    return None


_PAIRED = _find_paired_exposure()

pytestmark = pytest.mark.skipif(
    _PAIRED is None,
    reason=(
        "needs one exposure present both DAQ-native in "
        f"{REPLAY_DIR or '$DONUT_SERVER_REPLAY_DIR (unset)'} and assembled in "
        f"{RAW_DIR or '$DONUT_SERVER_RAW_DIR (unset)'}"
    ),
)


@pytest.fixture(scope="module")
def paired():
    directory, index = _PAIRED
    source, _ = daq.resolve_exposure_dir(directory)
    return source, index


@pytest.fixture(scope="module")
def reference_images(paired):
    """The assembled reference image plane per detector, read once."""
    import lsst.afw.image as afwImage

    source, index = paired
    return {
        det: afwImage.ExposureF.readFits(index[(source.visit, det)])
        for det in sorted(source.handles)
    }


def test_assembled_image_matches_the_reference(paired, reference_images):
    """Every detector, bit-for-bit, against a raw this code did not produce.

    This is the guard that lets the assembly internals be rewritten: it compares
    the *output* against an independently-produced file rather than asserting
    anything about how the amps were read.
    """
    source, _ = paired
    for det in sorted(source.handles):
        got = daq.load_raw(source.handles[det]).getImage().getArray()
        want = reference_images[det].getImage().getArray()
        assert got.shape == want.shape, f"detector {det} shape"
        assert np.array_equal(got, want), f"detector {det} pixels"


def test_bbox_matches_the_reference(paired, reference_images):
    source, _ = paired
    for det in sorted(source.handles):
        assert daq.load_raw(source.handles[det]).getBBox() == (
            reference_images[det].getBBox()
        ), f"detector {det}"


def test_mask_and_variance_are_zero(paired):
    """What `exposure_codec` relies on to not send them (its module docstring)."""
    source, _ = paired
    for det in sorted(source.handles):
        exposure = daq.load_raw(source.handles[det])
        assert not exposure.getMask().getArray().any(), f"detector {det} mask"
        assert not exposure.getVariance().getArray().any(), f"detector {det} variance"


def test_detector_and_amp_geometry_match_the_reference(paired, reference_images):
    source, _ = paired
    for det in sorted(source.handles):
        got = daq.load_raw(source.handles[det]).getDetector()
        want = reference_images[det].getDetector()
        assert (got.getName(), got.getId()) == (want.getName(), want.getId())
        assert [a.getRawBBox() for a in got] == [a.getRawBBox() for a in want]


def test_exposure_info_matches_the_reference(paired, reference_images):
    """WCS, VisitInfo and filter, which assembly attaches rather than reads."""
    import lsst.geom as geom

    source, _ = paired
    for det in sorted(source.handles):
        got = daq.load_raw(source.handles[det])
        want = reference_images[det]
        assert got.getFilter() == want.getFilter(), f"detector {det} filter"
        assert got.getInfo().getId() == want.getInfo().getId(), f"detector {det} id"

        # Compared by where they *map*, not with `==`. Both WCSs here are built
        # by createInitialSkyWcs from the same boresight, but one has been
        # through a FITS round trip, so they are unequal objects (and unequal
        # writeString()s) that agree to 0.0 arcsec everywhere. Same reason the
        # boresights below are compared with a tolerance rather than ==: they
        # differ by ~5e-11 arcsec of float noise.
        got_wcs, want_wcs = got.getWcs(), want.getWcs()
        bbox = want.getBBox()
        corners = list(bbox.getCorners()) + [geom.Point2I(bbox.getCenter())]
        for corner in corners:
            point = geom.Point2D(corner)
            separation = (
                got_wcs.pixelToSky(point)
                .separation(want_wcs.pixelToSky(point))
                .asArcseconds()
            )
            assert separation < 1e-6, f"detector {det} wcs at {point}: {separation}"

        got_visit = got.getInfo().getVisitInfo()
        want_visit = want.getInfo().getVisitInfo()
        boresight = (
            got_visit.getBoresightRaDec()
            .separation(want_visit.getBoresightRaDec())
            .asArcseconds()
        )
        assert boresight < 1e-6, f"detector {det} boresight: {boresight}"
