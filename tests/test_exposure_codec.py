"""Round-trip a real raw through the wire codec.

Skipped when `raw/` is empty, since the raws are large and not in version control.
"""
import glob
import os

import numpy as np
import pytest

from lsst.ts.donut_server import exposure_codec

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
RAW_PATHS = sorted(glob.glob(os.path.join(RAW_DIR, "raw_*.fits")))

pytestmark = pytest.mark.skipif(not RAW_PATHS, reason=f"no raw_*.fits in {RAW_DIR}")


@pytest.fixture(scope="module")
def original():
    import lsst.afw.image as afwImage

    return afwImage.ExposureF.readFits(RAW_PATHS[0])


@pytest.fixture(scope="module")
def decoded(original):
    img_blob, meta_blob = exposure_codec.encode_exposure(original)
    return exposure_codec.decode_exposure(img_blob, meta_blob)


def test_image_is_bit_identical(original, decoded):
    assert np.array_equal(decoded.getImage().getArray(), original.getImage().getArray())


def test_bbox_survives(original, decoded):
    # The metadata carrier is a 1x1 subimage, so the full bbox has to be sent
    # alongside it -- this is the regression guard for that.
    assert decoded.getBBox() == original.getBBox()


def test_mask_and_variance_are_zero(decoded):
    # Never sent over the wire; reconstructed by ExposureF(bbox) zero-filling.
    assert not decoded.getVariance().getArray().any()
    assert not decoded.getMask().getArray().any()


def test_detector_survives(original, decoded):
    got, want = decoded.getDetector(), original.getDetector()
    assert (got.getName(), got.getId()) == (want.getName(), want.getId())
    assert [a.getRawBBox() for a in got.getAmplifiers()] == [
        a.getRawBBox() for a in want.getAmplifiers()
    ]


def test_exposure_info_survives(original, decoded):
    assert decoded.getWcs() == original.getWcs()
    assert decoded.getInfo().getVisitInfo() == original.getInfo().getVisitInfo()
    assert decoded.getFilter() == original.getFilter()
    assert decoded.getInfo().getId() == original.getInfo().getId()


def test_metadata_keys_needed_for_dataids_survive(original, decoded):
    # build_quantum_context reads day_obs and group off these.
    for key in ("DAYOBS", "GROUPID", "OBSID"):
        assert decoded.getMetadata()[key] == original.getMetadata()[key]


def test_decode_rejects_wrong_sized_image(original):
    img_blob, meta_blob = exposure_codec.encode_exposure(original)
    with pytest.raises(ValueError, match="image blob is"):
        exposure_codec.decode_exposure(img_blob[:-4], meta_blob)


def test_decode_from_memoryview(original):
    """The service hands decode_exposure memoryviews onto shared memory, not bytes."""
    img_blob, meta_blob = exposure_codec.encode_exposure(original)
    block = bytearray(len(img_blob) + len(meta_blob))
    block[: len(img_blob)] = img_blob
    block[len(img_blob) :] = meta_blob

    view = memoryview(block)
    decoded = exposure_codec.decode_exposure(
        view[: len(img_blob)], view[len(img_blob) :]
    )
    assert np.array_equal(decoded.getImage().getArray(), original.getImage().getArray())
    assert decoded.getDetector().getName() == original.getDetector().getName()
