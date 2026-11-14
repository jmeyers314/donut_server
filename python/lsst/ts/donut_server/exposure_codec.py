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

"""Wire serialization of a raw `ExposureF`.

A raw corner-sensor exposure is ~114 MB on disk but only its **image** plane
carries information: the mask and variance planes of a raw are identically
zero. So we ship the float32 image verbatim (37.75 MB) plus a small pickle of
everything else, and let the receiver allocate the zero mask/variance itself.

`Detector`, `VisitInfo` and `FilterLabel` are individually *not* picklable,
but a whole `Exposure` is. So the metadata carrier is a 1x1 subimage. Its own
bbox is 1x1, so the full bbox has to be sent alongside it.

Deliberately not imported by `protocol.py`: that module stays stdlib-only so
the FastAPI process never needs afw.
"""

from __future__ import annotations

import pickle

import numpy as np

import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.ts.donut_server import protocol

IMG_DTYPE = np.float32


def encode_meta(exp: afwImage.ExposureF, bbox: geom.Box2I | None = None) -> bytes:
    """Just the metadata pickle for `exp` -- everything but the pixels.

    Pickling the full exposure is slow, but we can pickle the meta and pass
    the pixels as a buffer.

    Parameters
    ----------
    exp : lsst.afw.image.ExposureF
        The exposure to read metadata from.
    bbox : lsst.geom.Box2I, optional
        The image bbox to record. Defaults to `exp`'s own bbox.

    Returns
    -------
    bytes
        A pickle of the 1x1 metadata stub and the full bbox.

    Notes
    -----
    `exp` may be a whole exposure or a 1x1 stub carrying the same
    `ExposureInfo`. For a stub, `bbox` has to be given: it is what the
    receiver allocates from, and the stub's own bbox would rebuild a 1x1
    image.

    The stub's one pixel value is arbitrary and never read: `decode_exposure`
    fills every pixel from the image blob. Two pickles of the same raw can
    therefore differ byte-for-byte while decoding identically.
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

    Parameters
    ----------
    exp : lsst.afw.image.ExposureF
        A whole exposure. Its mask and variance planes are ignored: a
        raw's are identically zero, and `decode_exposure` reconstructs
        them by zero-fill.

    Returns
    -------
    img_blob : bytes
        The image plane alone, C-ordered.
    meta_blob : bytes
        Everything else, as `encode_meta` returns it.

    Raises
    ------
    ValueError
        If `exp`'s image plane is not `IMG_DTYPE`.
    """
    arr = exp.getImage().getArray()
    if arr.dtype != IMG_DTYPE:
        raise ValueError(f"expected {IMG_DTYPE} image, got {arr.dtype}")

    return arr.tobytes(), encode_meta(exp)


def decode_exposure(img_blob: protocol.Blob, meta_blob: protocol.Blob) -> afwImage.ExposureF:
    """Inverse of `encode_exposure`. Mask and variance come back all-zero.

    Parameters
    ----------
    img_blob : protocol.Blob
        The image plane. In the service this is a `memoryview` onto shared
        memory, so `np.frombuffer` below reads the pixels in place and the
        assignment into the `ExposureF` is the only copy.
    meta_blob : protocol.Blob
        The metadata pickle, as `encode_meta` returns it.

    Returns
    -------
    lsst.afw.image.ExposureF
        A full-size exposure, with mask and variance zero-filled.

    Raises
    ------
    ValueError
        If `img_blob`'s length disagrees with the bbox in `meta_blob`.
    """
    meta = pickle.loads(meta_blob)
    x0, y0, width, height = meta["bbox"]
    stub = meta["stub"]

    bbox = geom.Box2I(geom.Point2I(x0, y0), geom.Extent2I(width, height))
    exp = afwImage.ExposureF(bbox)

    expected = width * height * np.dtype(IMG_DTYPE).itemsize
    if len(img_blob) != expected:
        raise ValueError(f"image blob is {len(img_blob)} bytes, expected {expected} for {width}x{height}")
    exp.getImage().getArray()[:] = np.frombuffer(img_blob, dtype=IMG_DTYPE).reshape(height, width)

    exp.setDetector(stub.getDetector())
    exp.setWcs(stub.getWcs())
    exp.setFilter(stub.getFilter())
    exp.setMetadata(stub.getMetadata())
    exp.getInfo().setVisitInfo(stub.getInfo().getVisitInfo())
    exp.getInfo().setId(stub.getInfo().getId())
    return exp
