"""Wire (de)serialization of a raw `ExposureF`, shared by client and coordinator.

A raw corner-sensor exposure is ~114 MB on disk but only its **image** plane
carries information: the mask and variance planes of a raw are identically
zero. So we ship the float32 image verbatim (37.75 MB) plus a small pickle of
everything else, and let the receiver allocate the zero mask/variance itself --
`ExposureF(bbox)` already zero-fills all three planes.

`Detector`, `VisitInfo` and `FilterLabel` are individually *not* picklable, but
a whole `Exposure` is (afw routes it through an in-memory FITS). So the metadata
carrier is a 1x1 subimage: same `ExposureInfo`, negligible pixel payload. Its
own bbox is 1x1, so the full bbox has to be sent alongside it.

Deliberately not imported by `protocol.py`: that module stays stdlib-only so the
FastAPI process never needs afw.
"""
from __future__ import annotations

import pickle

import numpy as np

import lsst.afw.image as afwImage
import lsst.geom as geom

IMG_DTYPE = np.float32


def encode_meta(exp: afwImage.ExposureF, bbox=None) -> bytes:
    """Just the metadata pickle for `exp` -- everything but the pixels.

    Split out of `encode_exposure` for the producer's parallel path, where the
    pixels are written straight into shared memory and calling `encode_exposure`
    would mean a 37.75 MB `tobytes()` whose result is thrown away.

    `bbox` is the *full* image bbox to record, and must be given when `exp` is
    already a 1x1 stub rather than a whole exposure: it is what the receiver
    allocates from, so taking it from a stub's own bbox would rebuild a 1x1
    image. Defaults to `exp`'s bbox, which is correct for a whole exposure.

    Note that the stub's one pixel is *not* part of the contract: whether it
    holds a real pixel value (sliced from a whole exposure) or zero (a stub
    built fresh) makes the pickles differ byte-for-byte while decoding
    identically, because `decode_exposure` reads only the components off the
    stub and fills every pixel from the image blob. Compare decoded exposures,
    not these bytes.
    """
    full = exp.getBBox() if bbox is None else bbox
    stub = exp[geom.Box2I(exp.getBBox().getMin(), geom.Extent2I(1, 1))]
    return pickle.dumps(
        {
            "stub": stub,
            "bbox": (full.getMinX(), full.getMinY(), full.getWidth(), full.getHeight()),
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def encode_exposure(exp: afwImage.ExposureF) -> tuple[bytes, bytes]:
    """Split `exp` into (image bytes, metadata pickle).

    The image array is C-contiguous as read from disk, so `tobytes()` is a
    single memcpy.
    """
    arr = exp.getImage().getArray()
    if arr.dtype != IMG_DTYPE:
        raise ValueError(f"expected {IMG_DTYPE} image, got {arr.dtype}")

    return arr.tobytes(), encode_meta(exp)


def decode_exposure(img_blob, meta_blob) -> afwImage.ExposureF:
    """Inverse of `encode_exposure`. Mask and variance come back all-zero.

    Both arguments are any buffer -- in the service they are memoryviews onto
    shared memory, so `np.frombuffer` below reads the pixels in place and the
    assignment into the ExposureF is the only copy.
    """
    meta = pickle.loads(meta_blob)
    x0, y0, width, height = meta["bbox"]
    stub = meta["stub"]

    bbox = geom.Box2I(geom.Point2I(x0, y0), geom.Extent2I(width, height))
    exp = afwImage.ExposureF(bbox)

    expected = width * height * np.dtype(IMG_DTYPE).itemsize
    if len(img_blob) != expected:
        raise ValueError(
            f"image blob is {len(img_blob)} bytes, expected {expected} for {width}x{height}"
        )
    exp.getImage().getArray()[:] = np.frombuffer(img_blob, dtype=IMG_DTYPE).reshape(
        height, width
    )

    exp.setDetector(stub.getDetector())
    exp.setWcs(stub.getWcs())
    exp.setFilter(stub.getFilter())
    exp.setMetadata(stub.getMetadata())
    exp.getInfo().setVisitInfo(stub.getInfo().getVisitInfo())
    exp.getInfo().setId(stub.getInfo().getId())
    return exp
