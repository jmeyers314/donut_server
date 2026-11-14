# This file is part of donut_server.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Round-trip an exposure through the wire codec.

Every test runs twice: against a synthetic exposure built here, which needs
no data and so always runs, and against a real raw read off disk, which is
skipped unless DONUT_SERVER_RAW_DIR names a directory holding some.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.geom as geom
from lsst.obs.lsst import LsstCam
from lsst.ts.donut_server import exposure_codec

# Unset is a skip, not an error: read the variable directly rather than calling
# client.raw_dir(), which raises.
RAW_DIR = os.environ.get("DONUT_SERVER_RAW_DIR", "")
RAW_PATHS = sorted(glob.glob(os.path.join(RAW_DIR, "raw_*.fits"))) if RAW_DIR else []

needs_raw = pytest.mark.skipif(
    not RAW_PATHS,
    reason=f"no raw_*.fits in {RAW_DIR or '$DONUT_SERVER_RAW_DIR (unset)'}",
)

SYNTHETIC_DETECTOR = "R00_SW0"
SYNTHETIC_SHAPE = (32, 64)
SYNTHETIC_RA_DEG = 215.5
SYNTHETIC_DEC_DEG = -10.1
SYNTHETIC_MIDPOINT = "2026-07-13T00:00:50.000"


@pytest.fixture(scope="module")
def detector() -> cameraGeom.Detector:
    """A real LSSTCam corner-sensor detector, straight from cameraGeom.

    Returns
    -------
    lsst.afw.cameraGeom.Detector
        `SYNTHETIC_DETECTOR`, with all eight of its amplifiers.
    """
    return LsstCam.getCamera()[SYNTHETIC_DETECTOR]


def build_synthetic(detector: cameraGeom.Detector) -> afwImage.ExposureF:
    """An `ExposureF` carrying every component the codec moves.

    Built rather than read, so this file needs no data.

    Parameters
    ----------
    detector : lsst.afw.cameraGeom.Detector
        The detector to attach.

    Returns
    -------
    lsst.afw.image.ExposureF
        A float32 exposure with pseudo-random pixels, a detector, a WCS, a
        filter label, a header and a `VisitInfo`.
    """
    height, width = SYNTHETIC_SHAPE
    exp = afwImage.ExposureF(geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(width, height)))
    rng = np.random.default_rng(0)
    exp.getImage().getArray()[:] = rng.normal(size=SYNTHETIC_SHAPE).astype(exposure_codec.IMG_DTYPE)

    exp.setDetector(detector)
    exp.setWcs(
        afwGeom.makeSkyWcs(
            crpix=geom.Point2D(width / 2, height / 2),
            crval=geom.SpherePoint(SYNTHETIC_RA_DEG, SYNTHETIC_DEC_DEG, geom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=0.2 * geom.arcseconds),
        )
    )
    exp.setFilter(afwImage.FilterLabel(band="z", physical="z_20"))

    header = dafBase.PropertyList()
    header.set("DAYOBS", "20260713")
    header.set("GROUPID", SYNTHETIC_MIDPOINT)
    header.set("OBSID", "MC_O_20260713_000050")
    exp.setMetadata(header)

    exp.getInfo().setVisitInfo(
        afwImage.VisitInfo(
            exposureTime=30.0,
            date=dafBase.DateTime(SYNTHETIC_MIDPOINT, dafBase.DateTime.TAI),
            boresightRaDec=geom.SpherePoint(SYNTHETIC_RA_DEG, SYNTHETIC_DEC_DEG, geom.degrees),
        )
    )
    exp.getInfo().setId(2026071300050)
    return exp


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("synthetic", id="synthetic"),
        pytest.param("raw", id="raw", marks=needs_raw),
    ],
)
def original(request: pytest.FixtureRequest, detector: cameraGeom.Detector) -> afwImage.ExposureF:
    """The exposure under test, synthetic or real.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Carries the parameter naming which of the two sources to use.
    detector : lsst.afw.cameraGeom.Detector
        Attached to the synthetic exposure. Unused for a real raw, which
        brings its own.

    Returns
    -------
    lsst.afw.image.ExposureF
        Either `build_synthetic`'s exposure or the first raw in
        `RAW_PATHS`.
    """
    if request.param == "synthetic":
        return build_synthetic(detector)
    return afwImage.ExposureF.readFits(RAW_PATHS[0])


@pytest.fixture(scope="module")
def decoded(original: afwImage.ExposureF) -> afwImage.ExposureF:
    """`original` after one encode/decode round trip.

    Parameters
    ----------
    original : lsst.afw.image.ExposureF
        The exposure to round-trip.

    Returns
    -------
    lsst.afw.image.ExposureF
        The exposure rebuilt from the two wire blobs.
    """
    img_blob, meta_blob = exposure_codec.encode_exposure(original)
    return exposure_codec.decode_exposure(img_blob, meta_blob)


def test_image_is_bit_identical(original: afwImage.ExposureF, decoded: afwImage.ExposureF) -> None:
    """Pixels survive the round trip exactly, not within a tolerance."""
    assert np.array_equal(decoded.getImage().getArray(), original.getImage().getArray())


def test_bbox_survives(original: afwImage.ExposureF, decoded: afwImage.ExposureF) -> None:
    """The full bbox is recovered, not the metadata stub's 1x1 one.

    The metadata carrier is a 1x1 subimage, so the full bbox has to travel
    alongside it; this is the regression guard for that.
    """
    assert decoded.getBBox() == original.getBBox()


def test_mask_and_variance_are_zero(decoded: afwImage.ExposureF) -> None:
    """The two planes never sent are reconstructed by zero-fill.

    `ExposureF(bbox)` zero-fills all three planes, which is what lets the
    codec ship only the image.
    """
    assert not decoded.getVariance().getArray().any()
    assert not decoded.getMask().getArray().any()


def test_detector_survives(original: afwImage.ExposureF, decoded: afwImage.ExposureF) -> None:
    """Detector identity and amplifier geometry cross the wire intact."""
    got, want = decoded.getDetector(), original.getDetector()
    assert (got.getName(), got.getId()) == (want.getName(), want.getId())
    assert [a.getRawBBox() for a in got.getAmplifiers()] == [a.getRawBBox() for a in want.getAmplifiers()]


def test_exposure_info_survives(original: afwImage.ExposureF, decoded: afwImage.ExposureF) -> None:
    """The components that are individually unpicklable still arrive.

    `Detector`, `VisitInfo` and `FilterLabel` ride the 1x1 stub precisely
    because none of them pickles on its own.
    """
    assert decoded.getWcs() == original.getWcs()
    assert decoded.getInfo().getVisitInfo() == original.getInfo().getVisitInfo()
    assert decoded.getFilter() == original.getFilter()
    assert decoded.getInfo().getId() == original.getInfo().getId()


def test_metadata_keys_needed_for_dataids_survive(
    original: afwImage.ExposureF, decoded: afwImage.ExposureF
) -> None:
    """The header keys the butler builds data IDs from survive.

    `build_quantum_context` reads day_obs and group off these, so losing
    one would fail far downstream of the codec.
    """
    for key in ("DAYOBS", "GROUPID", "OBSID"):
        assert decoded.getMetadata()[key] == original.getMetadata()[key]


def test_no_header_key_is_lost(original: afwImage.ExposureF, decoded: afwImage.ExposureF) -> None:
    """Every header card arrives, except the ones the WCS absorbs.

    Stronger than naming individual keys, and the reason the round trip
    cannot simply be asserted metadata-equal: pickling the stub routes it
    through an in-memory FITS, which *adds* bookkeeping of its own --
    `TFORM1` and `TTYPE1` describe the binary table `ExposureInfo` is
    written as. Gains are therefore ignored and only losses checked.

    `RADESYS` is the one real loss: afw consumes it into the `SkyWcs` on
    read instead of leaving it in the header, so the information survives
    on the exposure, just not here. `test_exposure_info_survives` is what
    holds the WCS itself to account.
    """
    lost = set(original.getMetadata().names()) - set(decoded.getMetadata().names())
    assert lost <= {"RADESYS"}


def test_shared_header_values_are_unchanged(
    original: afwImage.ExposureF, decoded: afwImage.ExposureF
) -> None:
    """No card arrives holding a different value than it left with.

    The other half of `test_no_header_key_is_lost`: a key could survive
    the FITS trip while its value was reformatted or truncated, which is
    the failure that test alone would miss.
    """
    before, after = original.getMetadata(), decoded.getMetadata()
    shared = set(before.names()) & set(after.names())
    assert {key: after[key] for key in shared} == {key: before[key] for key in shared}


def test_decode_rejects_wrong_sized_image(original: afwImage.ExposureF) -> None:
    """A blob that disagrees with the recorded bbox raises, not reshapes."""
    img_blob, meta_blob = exposure_codec.encode_exposure(original)
    with pytest.raises(ValueError, match="image blob is"):
        exposure_codec.decode_exposure(img_blob[:-4], meta_blob)


def test_decode_from_memoryview(original: afwImage.ExposureF) -> None:
    """The service hands decode_exposure memoryviews onto shared memory.

    Not bytes -- so `protocol.Blob` has to cover the buffer case, and
    `np.frombuffer` has to read the pixels in place.
    """
    img_blob, meta_blob = exposure_codec.encode_exposure(original)
    block = bytearray(len(img_blob) + len(meta_blob))
    block[: len(img_blob)] = img_blob
    block[len(img_blob) :] = meta_blob

    view = memoryview(block)
    decoded = exposure_codec.decode_exposure(view[: len(img_blob)], view[len(img_blob) :])
    assert np.array_equal(decoded.getImage().getArray(), original.getImage().getArray())
    assert decoded.getDetector().getName() == original.getDetector().getName()
