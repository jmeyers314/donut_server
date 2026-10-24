"""Turn a directory of DAQ-native LSSTCam corner-sensor files into a `RawSource`.

**Temporary, in the same sense `scripts/run_sequence.py` says it is.** The
camera writes *unassembled amplifier* images -- one FITS per corner sensor with
eight `Segment1x` HDUs -- and the service's wire format carries an assembled
`ExposureF`. Assembly belongs on the producer side eventually (or the raws get
ingested into a butler repo and `donutClient.py -b` handles them); until then it
happens here, in-process, with `lsst.obs.lsst`.

This module is the shared half of two producers: the fixed-cadence walker in
`scripts/run_sequence.py` and the camera-triggered `trigger.py`. It was
originally the former's private code.

Layout it expects, one directory per exposure, as the replay tree is laid out:

    <dir>/MC_O_20260713_000050_R00_SW0.fits
         /MC_O_20260713_000050_R00_SW0.json          # not read; see trigger.py
         /MC_O_20260713_000050_R00_SG0_guider.fits   # ignored

Only `_R??_SW?.fits` is read: SG* are guider sensors, not corner wavefront
sensors, and the service has no calibs for them. (They also carry a *different*
`FILTER` than the SW files of the same exposure in the trees seen so far, so a looser glob
would trip the single-filter check below rather than failing quietly.)

### Substituted metadata

Every header read goes through `read_header`, and `resolve_exposure_dir` accepts
a `headers` map that replaces the on-disk primary header per detector. That one
seam is what lets a BTS replay -- which reproduces the pixels but not the
metadata -- be processed against the metadata the pixels were actually taken
with. See `trigger.py` for why that matters and where the map comes from.

Two rules make the substitution safe, and both are load-bearing:

- The map is keyed by **detector name**, and the name is read from the *live*
  file. Detector identity is a property of the pixels, so a substituted header
  is never what decides which detector we are looking at.
- `detector_num` likewise comes from the live header, because it indexes
  cameraGeom for the amp layout the pixels are in. Only the metadata that
  describes the *observation* -- pointing, filter, times, FOCUSZ -- is taken
  from the substitute.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any

from lsst.ts.donut_server import client
from lsst.ts.donut_server import exposure_codec

# The eight corner wavefront sensors, as DAQ filenames spell them. SW0 is
# extra-focal, SW1 intra-focal.
SW_PATTERN = "*_R??_SW?.fits"

EXPECTED_SENSORS = 8


@dataclass(frozen=True)
class RawHandle:
    """One sensor's file, plus the header to assemble it under.

    A `client.RawSource.handles` value. It carries the header rather than
    letting `load_raw` re-read it for two reasons: the header may be a
    *substitute* that is not on disk anywhere near `path`, and the non-replay
    path has already read it during resolution anyway, so re-reading is pure
    cost on the push critical path.

    `detector_num` is the live file's, never the header's -- see the module
    docstring.
    """

    path: str
    header: Any
    detector_num: int


def read_header(path: str):
    """The primary header, with `astro_metadata_translator`'s fixes applied.

    Fixed rather than raw because `ObservationInfo` is built from it and the
    service's own formatter path applies the same fixes -- an unfixed header is
    a different header. Headers handed back here are therefore already fixed,
    which is why substituting one needs no second pass.
    """
    from astro_metadata_translator import fix_header
    from lsst.afw.fits import readMetadata
    from lsst.obs.lsst.translators import LsstCamTranslator

    metadata = readMetadata(path, 0)
    fix_header(metadata, translator_class=LsstCamTranslator)
    return metadata


def obs_info(header, filename: str | None = None):
    """`ObservationInfo` for one sensor, from its primary header alone.

    Deliberately header-only: this is what makes the /prepare half of the cycle
    free of pixel reads. Eight of these cost ~0.1 s total, against ~0.1 s per
    sensor to assemble.
    """
    from astro_metadata_translator import ObservationInfo
    from lsst.obs.lsst.translators import LsstCamTranslator

    return ObservationInfo(
        header, translator_class=LsstCamTranslator, filename=filename
    )


def read_obs_info(path: str):
    """`ObservationInfo` straight off a file, for callers with no substitute."""
    return obs_info(read_header(path), filename=path)


def detector_name(header) -> str:
    """`R00_SW0`-style name, as the coordinator's calib dicts are keyed.

    Assembled from the header rather than parsed out of the filename: the
    filename is BTS's or the DAQ's to choose, whereas these two cards describe
    the hardware the pixels came off.
    """
    try:
        raft, slot = header["RAFTBAY"], header["CCDSLOT"]
    except (KeyError, LookupError) as exc:
        raise RuntimeError(
            f"header has no {exc} card, so its detector cannot be identified"
        ) from exc
    return f"{raft}_{slot}"


_CAMERA = None


def _camera():
    """`LsstCam.getCamera()`, cached across calls.

    This is the expensive part (~1 s) and it is identical for all 8 sensors.
    """
    global _CAMERA
    if _CAMERA is None:
        from lsst.obs.lsst import LsstCam

        _CAMERA = LsstCam.getCamera()
    return _CAMERA


def _assemble_image(handle: RawHandle, out=None):
    """Read one DAQ file's amp HDUs into a single untrimmed image array.

    What `readRawAmps` + `fixAmpsAndAssemble` do, minus the two thirds of the
    work that goes into planes nobody reads. Those helpers wrap every amp in a
    `MaskedImage` and then copy all three planes into the assembled CCD, but a
    raw's mask and variance are identically zero -- which is exactly why
    `exposure_codec` does not send them, and why the receiver reconstructs them
    with `ExposureF(bbox)`. Worth roughly a third of the serial encode.

    Verified bit-identical to `fixAmpsAndAssemble` on all eight corner sensors,
    against the independently-assembled `raw/raw_*.fits`; `tests/test_daq.py`
    is that comparison.

    `out` is an optional destination array to blit into, for a caller that has
    somewhere specific it wants the pixels and would otherwise copy them again.

    Returns `(array, bbox, detector)`, where `detector` has been through
    `makeUpdatedDetector` and so describes the *assembled* image: its amps'
    `getRawBBox()`es are in assembled coordinates, which is what the amps on an
    `ExposureF` read back from `raw/raw_*.fits` look like and what the receiver
    will compare against.
    """
    import numpy as np

    import lsst.afw.cameraGeom as cameraGeom
    import lsst.afw.cameraGeom.utils as cameraGeomUtils
    import lsst.afw.image as afwImage
    from lsst.obs.lsst.assembly import fixAmpGeometry, warn_once

    detector = _camera()[handle.detector_num]

    # `fixAmpGeometry` is kept even though this no longer calls
    # `fixAmpsAndAssemble`. It costs nothing measurable (what the rewrite saves
    # is the three-plane copy inside assembleCcd, not this check) and it is not
    # only a geometry *fix* but a guard: it raises when an on-disk bbox differs
    # from cameraGeom in a way overscan cannot explain. On today's files it
    # modifies 0 of 16 amps, so it is here entirely for the day the camera
    # starts writing a different layout -- which would otherwise land as
    # silently wrong pixels. Rebuilding the detector from its output is also
    # what `fixAmpsAndAssemble` does, and the geometry below has to come from
    # the corrected amps rather than the raw cameraGeom ones.
    with warn_once(handle.path) as logCmd:
        builder = detector.rebuild()
        builder.clear()
        readers = []
        for hdu, amp in enumerate(detector, start=1):
            reader = afwImage.ImageFitsReader(handle.path, hdu=hdu)
            readers.append(reader)
            fixed, _modified = fixAmpGeometry(
                amp,
                bbox=reader.readBBox(),
                metadata=reader.readMetadata(),
                logCmd=logCmd,
            )
            builder.append(fixed)
        detector = builder.finish()

    full = cameraGeomUtils.calcRawCcdBBox(detector)
    shape = (full.getHeight(), full.getWidth())
    if out is None:
        out = np.empty(shape, dtype=exposure_codec.IMG_DTYPE)
    elif out.shape != shape:
        raise RuntimeError(
            f"destination is {out.shape}, but {handle.path} assembles to {shape}"
        )

    for reader, amp in zip(readers, detector):
        array = reader.read(dtype=np.dtype(np.int32), allowUnsafe=True).getArray()
        if amp.getRawFlipX():
            array = array[:, ::-1]
        if amp.getRawFlipY():
            array = array[::-1, :]

        target = amp.getRawBBox()
        target.shift(amp.getRawXYOffset())
        out[
            target.getMinY() - full.getMinY() : target.getMaxY() + 1 - full.getMinY(),
            target.getMinX() - full.getMinX() : target.getMaxX() + 1 - full.getMinX(),
        ] = array

    # The amp images have been moved into place in one CCD-sized image, so the
    # amps have to be told where they now live -- exactly what assembleCcd does
    # for an untrimmed assembly, and what makes these bboxes match a detector
    # read back off an assembled file.
    return out, full, cameraGeom.makeUpdatedDetector(detector)


def _build_stub(handle: RawHandle, bbox, detector):
    """A 1x1 `ExposureF` carrying everything about this raw except its pixels.

    `exposure_codec` ships the metadata as a 1x1 subimage (see its module
    docstring for why), so the pixels this carries are never sent and a 1x1
    exposure is all the metadata path needs. `attachRawWcsFromBoresight`
    requires a real `Exposure` with metadata and a detector on it, which is why
    this cannot be a bare dict.

    `detector` is the assembled-coordinates one from `_assemble_image`, not raw
    cameraGeom: it is what crosses the wire, and the WCS is built against it.
    """
    import lsst.afw.image as afwImage
    import lsst.geom as geom
    from lsst.obs.lsst import LsstCam
    from lsst.obs.lsst.assembly import attachRawWcsFromBoresight

    stub = afwImage.ExposureF(geom.Box2I(bbox.getMin(), geom.Extent2I(1, 1)))
    stub.setDetector(detector)

    # Order matters: attachRawWcsFromBoresight re-reads the metadata off the
    # exposure to build the VisitInfo, so the header has to be on it first.
    # This is also what makes a substituted header reach the WCS: the pointing
    # it describes is the one the stars in these pixels were taken at, which is
    # what the coordinator's refcat-coverage cross-check is checking.
    stub.setMetadata(handle.header)
    attachRawWcsFromBoresight(stub)

    # The assembly helpers set VisitInfo, id and WCS but not the FilterLabel --
    # that is the formatter's job, and the coordinator cross-checks the pushed
    # filter against the one /prepare loaded calibs for, so an absent label
    # fails the push rather than silently pairing the wrong flat.
    physical = obs_info(handle.header, filename=handle.path).physical_filter
    stub.setFilter(
        afwImage.FilterLabel(
            band=LsstCam.filterDefinitions.physical_to_band[physical],
            physical=physical,
        )
    )
    return stub


def load_raw(handle: RawHandle):
    """Assemble one DAQ-native amplifier file into an `ExposureF`.

    This reproduces what `LsstCamRawFormatter.readFull()` does for a whole-CCD
    read, minus the butler: read the per-amp HDUs, reconcile cameraGeom's
    overscan regions against the on-disk bboxes, assemble untrimmed, then
    attach the components that live in the header. The result is byte-for-byte
    the shape `raw/raw_*.fits` already has -- a 4608x2048 float32 image with
    overscan present and ISR not yet run -- which is what the coordinator's
    `runQuantum` expects to be handed.

    The whole-exposure entry point, for `RawSource.load` and anything else
    wanting one object; `_assemble_image` and `_build_stub` are the two halves
    for a caller that needs the pixels and the metadata separately.

    Imported lazily, so that importing this module costs nothing.
    """
    import lsst.afw.image as afwImage

    array, bbox, detector = _assemble_image(handle)
    stub = _build_stub(handle, bbox, detector)

    exposure = afwImage.ExposureF(bbox)
    exposure.getImage().getArray()[:] = array
    exposure.setDetector(stub.getDetector())
    exposure.setMetadata(stub.getMetadata())
    exposure.setWcs(stub.getWcs())
    exposure.setFilter(stub.getFilter())
    exposure.getInfo().setVisitInfo(stub.getInfo().getVisitInfo())
    exposure.getInfo().setId(stub.getInfo().getId())
    return exposure


def sw_paths(directory: str) -> list[str]:
    """The corner-sensor files in one exposure directory, sorted.

    An empty result names the pattern, and says so differently when the only
    thing present is compressed: whether our step in the camera's per-file
    script runs before or after its `fpack` step is not yet settled, and
    `readRawAmps` cannot read a `.fz` anyway, so guessing would turn a wiring
    question into an obscure stack traceback.
    """
    paths = sorted(glob.glob(os.path.join(directory, SW_PATTERN)))
    if paths:
        return paths
    if glob.glob(os.path.join(directory, SW_PATTERN + ".fz")):
        raise RuntimeError(
            f"{directory!r} holds only fpacked {SW_PATTERN}.fz files; this "
            "producer reads uncompressed DAQ files, so it must run before the "
            "compression step (or be taught to funpack)"
        )
    raise RuntimeError(f"no {SW_PATTERN} files in {directory!r}")


def resolve_exposure_dir(
    directory: str, headers: dict[str, Any] | None = None
) -> tuple[client.RawSource, str]:
    """Build a `RawSource` for one exposure directory, reading no pixels.

    Returns the source and the exposure's `IMGTYPE`, which callers report so an
    off-nominal frame is identifiable in the log.

    The pointing and the calib time come off the headers here rather than from
    a raw's `VisitInfo` later, so `client.read_prepare_args` finds both already
    set and /prepare goes out without opening a single amplifier HDU.
    `calib_time` is the exposure *midpoint*, matching what the butler and files
    paths in `client.py` both resolve calibs against (verified equal to the
    assembled `VisitInfo.date` to the millisecond).

    `headers`, when given, maps detector name -> primary header and replaces
    what is on disk for that detector. Every detector found must be in it: a
    partial map would mean half the exposure resolved against one pointing and
    half against another, which is worse than not substituting at all.
    """
    paths = sw_paths(directory)

    handles: dict[int, RawHandle] = {}
    filters: set[str] = set()
    exposure_ids: set[int] = set()
    image_types: set[str] = set()
    pointings: list[tuple[float, float]] = []
    midpoints: list[str] = []

    for path in paths:
        live = read_header(path)
        name = detector_name(live)
        # The live header still decides which detector this is, and so which
        # amp layout the pixels are in; only the observation metadata can be
        # substituted. See the module docstring.
        detector_num = obs_info(live, filename=path).detector_num

        header = live
        if headers is not None:
            if name not in headers:
                raise RuntimeError(
                    f"no substitute header for detector {name} "
                    f"({os.path.basename(path)}); have "
                    f"{sorted(headers)}"
                )
            header = headers[name]

        info = obs_info(header, filename=path)
        handles[detector_num] = RawHandle(path, header, detector_num)
        filters.add(str(info.physical_filter))
        exposure_ids.add(int(info.exposure_id))
        image_types.add(str(info.observation_type))
        if info.tracking_radec is None:
            raise RuntimeError(
                f"{os.path.basename(path)} has no tracking_radec; its RA/DEC "
                "headers are unpopulated, so no WCS can be built for it"
            )
        pointings.append(
            (info.tracking_radec.ra.deg, info.tracking_radec.dec.deg)
        )
        begin, end = info.datetime_begin, info.datetime_end
        midpoints.append((begin + (end - begin) / 2).tai.isot)

    # Each of these would otherwise surface much later and much less legibly:
    # a mixed filter as a coordinator cross-check failure mid-push, a mixed
    # exposure_id as "raws span multiple visits".
    if len(filters) != 1:
        raise RuntimeError(f"{directory!r} spans physical_filters {sorted(filters)}")
    if len(exposure_ids) != 1:
        raise RuntimeError(f"{directory!r} spans exposures {sorted(exposure_ids)}")
    if len(handles) != EXPECTED_SENSORS:
        raise RuntimeError(
            f"{directory!r} has {len(handles)} corner sensor(s), expected "
            f"{EXPECTED_SENSORS}: detectors {sorted(handles)}"
        )

    (physical_filter,) = filters
    (exposure_id,) = exposure_ids
    source = client.RawSource(
        visit=exposure_id,
        # Rejects a filter there are no calibs for here, at discovery, rather
        # than as a missing-flat error deep in the coordinator's calib load.
        physical_filter=client._check_known_filter(exposure_id, physical_filter),
        handles=handles,
        load=load_raw,
        boresight=pointings[0],
        calib_time=min(midpoints),
    )
    return source, "/".join(sorted(image_types))
