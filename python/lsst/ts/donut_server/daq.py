"""Turn a directory of DAQ-native LSSTCam corner-sensor files into a `RawSource`.

The camera writes *unassembled amplifier* images -- one FITS per corner sensor
with eight `Segment1x` HDUs -- so something has to assemble them into the
whole-CCD `ExposureF` the task expects. That is this module, in-process, with
`lsst.obs.lsst`.

This is a distinct path, not a stopgap for one. The alternative shapes are a
producer that ships assembled exposures (`client.resolve_from_files`) and a
butler that has already ingested them (`client.resolve_from_butler`, i.e.
`donutClient.py -b`); all three exist, and this is the one the camera actually
drives. Which is why it is the one that gets the fast machinery: parallel
assembly, and `ingested_exposures` for the server-side path where there is no
wire format at all.

Three callers, on both sides of the wire: the fixed-cadence walker in
`scripts/run_sequence.py` and the camera-triggered `trigger.py` both assemble
here and push the pixels, while the coordinator's `/ingest` reads the directory
itself and skips the push entirely. It was originally `run_sequence.py`'s
private code.

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

import os

# Before importing anything that reaches numpy: keep BLAS/OpenMP
# single-threaded, because `build_parts_parallel` forks. This is for fork
# safety, not speed -- A/B'd on the serial assembly loop at 1.92 s either way --
# but this numpy links OpenBLAS built with USE_OPENMP, and forking with live
# OpenMP threads is undefined behaviour.
#
# It has to be *here*, above the `client` import: that reaches
# `exposure_codec` and so afw and numpy, and these only bind if they are set
# before numpy loads. `bin.src/donutTrigger.py` sets the same five for the
# camera's hook; `setdefault` makes the overlap harmless, and this copy is what
# covers `scripts/run_sequence.py` and any other in-process importer.
#
# Measured, and the reason `_assert_single_threaded_blas` below checks rather
# than trusts: without this block, resolving one exposure directory takes the
# process from 1 to 18 OS tasks (17 live OpenMP threads) before any fork, while
# `threading.active_count()` still reports 1.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import contextlib
import glob
import multiprocessing as mp
import signal
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

from lsst.ts.donut_server import client
from lsst.ts.donut_server import exposure_codec
from lsst.ts.donut_server import protocol

# The eight corner wavefront sensors, as DAQ filenames spell them. SW0 is
# extra-focal, SW1 intra-focal.
SW_PATTERN = "*_R??_SW?.fits"

EXPECTED_SENSORS = 8

# Forced-serial escape hatch, for A/B measurement on a production box and to
# have something to reach for if the fork pool ever misbehaves in the field.
SERIAL_ENCODE_VAR = "DONUT_SERVER_SERIAL_ENCODE"

# How long to wait for the eight workers before concluding one died without
# reporting. Deliberately generous against how long they actually take --
# this is the backstop for a worker that is *gone*, not a cadence control,
# and the serial fallback it triggers is not free either. See `_fan_out`.
FAN_OUT_TIMEOUT_S = 120.0


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


# A corner sensor to hang the warm-up stub on. Any detector would do -- the
# translator discovery it provokes is not detector-specific -- so this is R00_SW0
# simply because it is the one the camera's hook fires on.
_WARMUP_DETECTOR = 191

# Enough of an LSSTCam primary header for `ObservationInfo` to pick a translator
# and build a VisitInfo from. Not a realistic exposure and not meant to be: the
# only thing wanted from it is the side effect. See `warm_up`.
_WARMUP_CARDS = (
    ("INSTRUME", "LSSTCam"),
    ("RA", 0.0),
    ("DEC", 0.0),
    ("ROTPA", 0.0),
    ("DATE-BEG", "2026-01-01T00:00:00.000"),
    ("DATE-END", "2026-01-01T00:00:30.000"),
    ("RAFTBAY", "R00"),
    ("CCDSLOT", "SW0"),
    ("OBSTYPE", "science"),
    ("EXPTIME", 30.0),
)


def warm_up() -> None:
    """Pay the one-time costs of the assembly path now, not on the first exposure.

    Two of them, both measured, both otherwise landing on whichever exposure
    happens to arrive first:

    - `LsstCam.getCamera()`, ~0.4 s.
    - the first `attachRawWcsFromBoresight` call, ~0.66 s against ~0.007 s for
      every call after it. The cost is `astro_metadata_translator` discovering
      which translator class a header belongs to, and it is per *process* -- so
      without this, all eight fork workers pay it simultaneously on their first
      call (0.51 s each, measured) rather than inheriting the result.

    A synthetic header is enough to trigger the discovery, and a real one
    immediately afterwards takes 0.007 s -- verified. The resulting exposure is
    nonsense and is discarded; `attachRawWcsFromBoresight` may well decline to
    attach a WCS at all, which is fine, since what is wanted is the lookup it
    does on the way there.

    Cheap to call more than once: `_camera` is cached and the translator
    discovery is idempotent.

    Logging is suppressed for the duration. The synthetic header is missing
    cards a real one has (`LSST_NUM`, dark time), and the translator says so at
    WARNING -- which on an otherwise clean startup reads like a real problem
    with the service rather than an artifact of a deliberately minimal header.
    """
    import logging as _logging

    import lsst.afw.image as afwImage
    import lsst.daf.base as dafBase
    import lsst.geom as geom
    from lsst.obs.lsst.assembly import attachRawWcsFromBoresight

    camera = _camera()

    header = dafBase.PropertyList()
    for card, value in _WARMUP_CARDS:
        header.set(card, value)

    stub = afwImage.ExposureF(geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(1, 1)))
    stub.setDetector(camera[_WARMUP_DETECTOR])
    stub.setMetadata(header)

    previous = _logging.root.manager.disable
    _logging.disable(_logging.CRITICAL)
    try:
        attachRawWcsFromBoresight(stub)
    finally:
        _logging.disable(previous)


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

    `out` is an optional destination array to blit into -- in the parallel path
    it is a view onto shared memory, so the assembled image is never
    materialized in this process's heap at all.

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

    Still the whole-exposure entry point, for `RawSource.load` and anything
    wanting one object. The push path does not use it: it wants the pixels in
    shared memory and the metadata as a pickle, and never needs the two
    assembled into an `ExposureF` on this side of the wire.

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


def _assert_single_threaded_blas() -> None:
    """Check that the env block at the top of this module actually bound.

    Deliberately `threadpoolctl` and not `threading.active_count()`: the threads
    that make forking unsafe here are native OpenMP ones, which the Python-level
    count cannot see. Measured on this path -- resolving one exposure directory
    and loading the camera takes the process to 18 entries in /proc/self/task
    while `active_count()` still says 1, so an `active_count()` assert would
    wave exactly this situation through. `coordinator.py` asserts both because
    its own fork workers can also inherit Python threads; here the env block is
    the only thing standing between us and forking with 17 live OpenMP threads.
    """
    import threadpoolctl

    hot = {
        info["prefix"]: info["num_threads"]
        for info in threadpoolctl.threadpool_info()
        if info["num_threads"] != 1
    }
    if hot:
        raise RuntimeError(
            f"thread pools not clamped: {hot}. Something imported numpy before "
            "lsst.ts.donut_server.daq, so the thread-limit env vars never "
            "bound, and forking with live OpenMP threads is undefined "
            "behaviour. Export them before starting the process (see "
            "bin.src/donutTrigger.py)."
        )


def _child_init() -> None:
    """Make a fork worker safe to be one. Runs once per child, before any task.

    `trigger.arm_watchdog` installs a SIGALRM handler that raises `TimeoutError`
    from whatever frame happens to be executing. A fork clears the pending alarm
    but *inherits the handler*, so without this a child could raise out of the
    middle of `multiprocessing`'s own queue machinery. The parent keeps its
    watchdog; the children must not have one, since they are bounded by the
    parent's pool teardown instead.
    """
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


# Set by `_fan_out` in the parent before forking, read by the children through
# the inherited address space. The block name and slot table are a few hundred
# bytes, but routing them through the pool's pickle channel per task would mean
# re-opening the block on every call rather than once per child.
_WORKER_CONTEXT: dict[str, Any] = {}


def _worker_open_block() -> Any:
    """This child's mapping of the parent's block, opened once and cached.

    `track=False` because the parent created the block and is the only process
    that may unlink it -- without it this child's resource_tracker would unlink
    it on exit and the parent would be left holding a name that no longer
    resolves. Same reason, same flag, as the coordinator's own attach.
    """
    block = _WORKER_CONTEXT.get("block")
    if block is None:
        block = shared_memory.SharedMemory(
            name=_WORKER_CONTEXT["shm_name"], track=False
        )
        _WORKER_CONTEXT["block"] = block
    return block


def _encode_one(det_id: int):
    """Write one detector's pixels into the shared block; return its meta pickle.

    Runs in a fork worker. The pixels -- 37.75 MB -- go straight into the block
    at their final wire offset, so they are never pickled and the parent never
    concatenates them. The meta pickle is ~827 kB and comes back the ordinary
    way, which measures as a rounding error and is not worth a second region
    in the layout.

    Errors come back as a value rather than an exception, so one unreadable
    sensor names itself instead of surfacing as a `multiprocessing` traceback
    with no detector in it.
    """
    try:
        import numpy as np

        handle = _WORKER_CONTEXT["handles"][det_id]
        offset, byte_count, shape = _WORKER_CONTEXT["slots"][det_id]

        expected = shape[0] * shape[1] * np.dtype(exposure_codec.IMG_DTYPE).itemsize
        if expected != byte_count:
            raise RuntimeError(
                f"detector {det_id} slot is {byte_count} bytes but its shape "
                f"{shape} needs {expected}"
            )

        block = _worker_open_block()
        view = memoryview(block.buf)
        try:
            # The assembled image is written *through* this view into the
            # parent's block, so it is never materialized in this child's heap
            # and never pickled back.
            target = np.ndarray(
                shape, dtype=exposure_codec.IMG_DTYPE, buffer=view, offset=offset
            )
            _array, bbox, detector = _assemble_image(handle, out=target)
            del target, _array
        finally:
            view.release()

        # `bbox` explicitly: the stub is 1x1, and it is this bbox the receiver
        # allocates the full image from.
        meta = exposure_codec.encode_meta(
            _build_stub(handle, bbox, detector), bbox=bbox
        )
        return det_id, detector.getName(), meta, None
    except Exception as exc:  # noqa: BLE001 -- reported, not raised; see docstring
        return det_id, None, None, f"{type(exc).__name__}: {exc}"


def _assemble_one(det_id: int):
    """Write one detector's pixels into the shared block. Returns no pixels.

    `_encode_one`'s sibling for the server-side ingest path, and deliberately
    thinner: it returns only the detector id, because the consumer is in the
    same process as the parent and can build the metadata itself from the header
    it already holds. `_encode_one` has to pickle a ~827 kB stub back because its
    consumer is across an HTTP boundary; here that would cost 0.037 s per
    detector to *unpickle* against 0.0135 s to rebuild, so this does not send one.

    Errors come back as a value for the same reason as `_encode_one`'s: one
    unreadable sensor should name itself rather than surface as a
    `multiprocessing` traceback with no detector in it.
    """
    try:
        import numpy as np

        handle = _WORKER_CONTEXT["handles"][det_id]
        offset, byte_count, shape = _WORKER_CONTEXT["slots"][det_id]

        expected = shape[0] * shape[1] * np.dtype(exposure_codec.IMG_DTYPE).itemsize
        if expected != byte_count:
            raise RuntimeError(
                f"detector {det_id} slot is {byte_count} bytes but its shape "
                f"{shape} needs {expected}"
            )

        block = _worker_open_block()
        view = memoryview(block.buf)
        try:
            target = np.ndarray(
                shape, dtype=exposure_codec.IMG_DTYPE, buffer=view, offset=offset
            )
            _assemble_image(handle, out=target)
            del target
        finally:
            view.release()

        return det_id, None, None, None
    except Exception as exc:  # noqa: BLE001 -- reported, not raised; see docstring
        return det_id, None, None, f"{type(exc).__name__}: {exc}"


def _image_shape(detector) -> tuple[int, int]:
    """The `(height, width)` one assembled detector's image plane will have.

    From cameraGeom alone, so the parent can lay the whole blob out before any
    pixel is read -- which is what lets the children write at final wire
    offsets. `calcRawCcdBBox` rather than `detector.getBBox()` because what is
    being sent is the *untrimmed* image, overscan included.

    A detector whose on-disk bboxes disagree with cameraGeom would make this
    wrong, so the worker re-checks its slot against the shape it actually
    assembled (and `fixAmpGeometry` would have raised first).
    """
    import lsst.afw.cameraGeom.utils as cameraGeomUtils

    bbox = cameraGeomUtils.calcRawCcdBBox(detector)
    return bbox.getHeight(), bbox.getWidth()


class ParallelEncodeUnavailable(RuntimeError):
    """The parallel path cannot run, and the caller should go serial.

    Its own type so `client.run_once` can tell "this machine or this exposure
    cannot do it" (fall back, log, carry on) from a genuine bug in the fan-out
    (let it propagate).
    """


def build_parts_parallel(source, n_workers: int | None = None):
    """Encode one exposure into a shared block, eight detectors at a time.

    Returns `(block, view, total_bytes)` -- the caller POSTs `view[:total_bytes]`
    and is responsible for releasing the view and closing *and unlinking* the
    block when the push is done. `encoded_blob` is the context manager that does
    that; prefer it.

    The serial path this replaces reads, assembles and encodes each sensor in
    turn and then concatenates the lot, which is almost entirely cfitsio
    parsing and per-amp array handling rather than I/O -- cold and warm
    re-reads differ little, and a bare read of one amp file is a rounding error
    against parsing it. Threads do not help, because the amp reads hold the GIL
    and a thread pool stops scaling after about two, so this forks instead.
    `scripts/bench_encode.py` measures all of this on whatever machine it is
    asked about; the ratios travel between machines and the seconds do not.

    Two things make the fork cheap. The children inherit the `LsstCam` camera
    through the address space (so `_camera()` is warmed below, before the pool
    exists, rather than in all eight children), and they return only the small
    meta pickle, because the pixels are already in the block the parent will
    send.
    """
    if os.environ.get(SERIAL_ENCODE_VAR):
        raise ParallelEncodeUnavailable(f"{SERIAL_ENCODE_VAR} is set")

    det_ids = sorted(source.handles)
    if not det_ids:
        raise ParallelEncodeUnavailable("no detectors to encode")

    # Both before the fork, and in this order: the camera is what the children
    # need inherited, and the assert is only meaningful once everything that
    # might start a thread pool has been imported.
    camera = _camera()
    try:
        _assert_single_threaded_blas()
    except RuntimeError as exc:
        # Reported as "cannot parallelize" rather than allowed to propagate: an
        # unclamped process is a deployment problem, but the serial encoder does
        # not fork and so is perfectly safe in one. Falling back keeps the
        # exposure and still says why on stdout.
        raise ParallelEncodeUnavailable(str(exc)) from exc

    import numpy as np

    itemsize = np.dtype(exposure_codec.IMG_DTYPE).itemsize
    names = {det_id: detector_name(source.handles[det_id].header) for det_id in det_ids}
    shapes = {
        det_id: _image_shape(camera[source.handles[det_id].detector_num])
        for det_id in det_ids
    }
    img_sizes = {
        det_id: shapes[det_id][0] * shapes[det_id][1] * itemsize for det_id in det_ids
    }

    # The image parts are laid out first, at offsets known now; the meta parts
    # go after them, at offsets known only once the children have returned. A
    # fixed slot per meta would also work (they come out the same size on every
    # exposure tested) but pinning a size the codec owns would be a trap the
    # first time it changes, so instead the framing is built once, at the end,
    # from the real sizes.
    sizes: dict[str, int] = {}
    offsets: dict[str, int] = {}
    cursor = 0
    for det_id in det_ids:
        part = protocol.part_name(names[det_id], protocol.IMG_SUFFIX)
        sizes[part] = img_sizes[det_id]
        offsets[part] = cursor
        cursor += img_sizes[det_id]
    img_payload_end = cursor

    # Sized for the images plus a generous allowance for the eight meta parts,
    # whose exact sizes are not known yet. The block is written sparsely and
    # tmpfs faults pages in lazily, so an allowance that is never touched costs
    # nothing; what it buys is one allocation rather than a resize.
    meta_allowance = EXPECTED_SENSORS * 4 * 1024 * 1024
    descriptors = protocol.descriptor_region_size(
        [
            protocol.part_name(names[det_id], suffix)
            for det_id in det_ids
            for suffix in (protocol.IMG_SUFFIX, protocol.META_SUFFIX)
        ]
    )
    block_size = protocol.HEADER_SIZE + descriptors + img_payload_end + meta_allowance

    # Checked before the block is written, not after: SharedMemory(create=True)
    # is only shm_open + ftruncate, which reserves nothing on tmpfs and succeeds
    # however small the mount is. The shortfall then surfaces when a write
    # touches a page tmpfs cannot allocate, and the kernel's only recourse is
    # SIGBUS -- the process dies mid-push with nothing but "Bus error". Same
    # trap, same reasoning, as the server's own capacity check.
    free = protocol.shm_free_bytes()
    if free is not None and free < block_size:
        raise ParallelEncodeUnavailable(
            f"{protocol.SHM_MOUNT} has {free / 1024 / 1024:.0f} MB free but this "
            f"exposure needs {block_size / 1024 / 1024:.0f} MB"
        )

    try:
        block = shared_memory.SharedMemory(create=True, size=block_size)
    except OSError as exc:
        raise ParallelEncodeUnavailable(f"cannot create shared block: {exc}") from exc

    view = None
    try:
        data_start = protocol.HEADER_SIZE + descriptors
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT.update(
            shm_name=block.name,
            handles=dict(source.handles),
            slots={
                det_id: (
                    data_start
                    + offsets[protocol.part_name(names[det_id], protocol.IMG_SUFFIX)],
                    img_sizes[det_id],
                    shapes[det_id],
                )
                for det_id in det_ids
            },
        )

        metas = _fan_out(det_ids, n_workers)

        view = memoryview(block.buf)

        # Now that the meta sizes are real, finish the layout and write the
        # framing. Gaps are legal in this format, but there are none here: the
        # metas are packed directly after the last image.
        cursor = img_payload_end
        for det_id in det_ids:
            part = protocol.part_name(names[det_id], protocol.META_SUFFIX)
            sizes[part] = len(metas[det_id])
            offsets[part] = cursor
            cursor += len(metas[det_id])

        framing, absolute, total = protocol.build_framing(sizes, offsets)
        if total > block_size:
            raise ParallelEncodeUnavailable(
                f"metadata came to {total - block_size} bytes more than the "
                f"{meta_allowance} byte allowance"
            )

        view[: len(framing)] = framing
        for det_id in det_ids:
            part = protocol.part_name(names[det_id], protocol.META_SUFFIX)
            start = absolute[part]
            view[start : start + len(metas[det_id])] = metas[det_id]

        return block, view, total
    except Exception:
        # The block is this function's to clean up until it has handed it back.
        if view is not None:
            view.release()
        block.close()
        block.unlink()
        raise
    finally:
        _WORKER_CONTEXT.clear()


def _fan_out(
    det_ids: list[int],
    n_workers: int | None,
    timeout: float | None = None,
    worker=None,
):
    """Run `worker` over `det_ids` in a fork pool. Returns {det_id: meta}.

    `worker` is `_encode_one` for the push path and `_assemble_one` for ingest;
    both write their pixels into the shared block and differ only in what they
    return. It must be a module-level function, since the pool pickles it by
    qualified name.

    `fork`, not `spawn`: the whole point is that the children inherit the camera
    and the loaded stack. That is safe here and not in the front end because
    this process is single-threaded by the time it gets here -- which is what
    the env block at the top of this module and `_assert_single_threaded_blas`
    together guarantee.

    `timeout` is not belt-and-braces: `Pool.map` waits on a result queue, so a
    worker that dies *without returning* -- SIGKILL from the OOM killer, a
    segfault in cfitsio, a stray killpg -- leaves it blocked forever (verified:
    a SIGKILLed worker hangs `map` indefinitely). An exception in a worker is
    fine, and both workers convert those to return values anyway; it is only
    death that strands the parent. Nothing supervises a detached trigger, so
    waiting forever means an exposure that never completes and never says why.
    """
    # Read at call time, not bound as a default, so the module global stays the
    # one place to change it (and tests can shorten it).
    if timeout is None:
        timeout = FAN_OUT_TIMEOUT_S
    if worker is None:
        worker = _encode_one

    workers = min(n_workers or len(det_ids), len(det_ids))
    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers, initializer=_child_init)
    try:
        results = pool.map_async(worker, det_ids).get(timeout=timeout)
    except mp.TimeoutError as exc:
        raise ParallelEncodeUnavailable(
            f"a worker did not return within {timeout:g}s; assuming one died "
            "without reporting (OOM killer, segfault, or external kill)"
        ) from exc
    except Exception as exc:
        raise ParallelEncodeUnavailable(f"fork pool failed: {exc}") from exc
    finally:
        # terminate() before join(), always: if the parent's own SIGALRM
        # watchdog fired while we were in map(), a bare join() on a pool with
        # live children would block until that alarm's TimeoutError had nowhere
        # left to go. Nothing supervises a detached trigger, so a child left
        # running here would be a permanent orphan holding a core. This is also
        # what reaps the survivors when the timeout above fires.
        pool.terminate()
        pool.join()

    metas = {}
    failures = []
    for det_id, _name, meta, error in results:
        if error is not None:
            failures.append(f"detector {det_id}: {error}")
        else:
            metas[det_id] = meta
    if failures:
        raise ParallelEncodeUnavailable("; ".join(failures))
    return metas


@contextlib.contextmanager
def encoded_blob(source, n_workers: int | None = None):
    """The push body for `source`, as a buffer the caller can POST directly.

    Yields a `memoryview` of exactly the blob. Releasing the view and unlinking
    the block are this manager's business, in that order: a block cannot be
    closed while any view onto it is still exported, and the creator is the only
    process that may unlink.

    Falls back to the serial encoder on anything that means "not here, not now"
    -- a small /dev/shm, a child that died, `DONUT_SERVER_SERIAL_ENCODE` -- so a
    transient shared-memory problem costs ~1.3 s rather than the exposure. A
    bug in the fan-out itself is not caught here and will propagate.
    """
    try:
        block, view, total = build_parts_parallel(source, n_workers)
    except ParallelEncodeUnavailable as exc:
        print(f"encode: parallel path unavailable ({exc}); using serial")
        yield memoryview(protocol.pack_blob(client.build_raw_parts(source)))
        return

    # Sliced once, here, so there is a name to release. A `yield view[:total]`
    # would hand out a second exported view that nothing holds a reference to,
    # and `close()` below would then raise BufferError -- the block cannot be
    # closed while *any* view onto it is live, not just the one we named.
    body = view[:total]
    try:
        yield body
    finally:
        body.release()
        view.release()
        block.close()
        try:
            block.unlink()
        except FileNotFoundError:
            # Already gone; nothing to reclaim and nothing to report.
            pass


@contextlib.contextmanager
def ingested_exposures(source, n_workers: int | None = None):
    """`{detector_name: ExposureF}` for `source`, assembled in this process.

    The server-side counterpart to `encoded_blob`. Where that one produces a
    wire blob for a producer to POST, this produces the exposures themselves --
    so when the consumer is in this process there is no wire format, no pickled
    pixels and no decode step at all.

    Three things make it cheap, all measured:

    - Eight forked workers assemble straight into one shared block, as the push
      path's fan-out already does.
    - The exposures are **views onto that block**, not copies:
      `ExposureF(MaskedImageF(ImageF(array=..., deep=False)))` shares the
      pixels. (`ExposureF(bbox)` then `setImage` does *not* share -- that route
      copies.)
    - The metadata is rebuilt here rather than returned by the children. They
      hold the header already, and `makeUpdatedDetector` + `calcRawCcdBBox` give
      the geometry with no pixels read, so a stub costs 0.0135 s to build
      against 0.037 s merely to unpickle one.

    A context manager because the exposures alias the block: it must outlive
    them, and every view onto it has to be released before it can be closed.
    The caller must therefore be done with the exposures when the `with` exits.

    Unlike `encoded_blob` there is no serial fallback. The caller is the
    coordinator, which has no second way to get pixels, so a failure here is a
    failed job -- `ParallelEncodeUnavailable` propagates and the command replies
    with it.
    """
    import numpy as np

    import lsst.afw.cameraGeom as cameraGeom
    import lsst.afw.cameraGeom.utils as cameraGeomUtils
    import lsst.afw.image as afwImage

    det_ids = sorted(source.handles)
    if not det_ids:
        raise ParallelEncodeUnavailable("no detectors to assemble")

    camera = _camera()
    try:
        _assert_single_threaded_blas()
    except RuntimeError as exc:
        raise ParallelEncodeUnavailable(str(exc)) from exc

    itemsize = np.dtype(exposure_codec.IMG_DTYPE).itemsize
    shapes = {
        det_id: _image_shape(camera[source.handles[det_id].detector_num])
        for det_id in det_ids
    }

    # One slot per detector, back to back. No framing and no descriptors: this
    # block is never sent anywhere, so the offsets only have to agree between
    # the workers that write them and the wraps below that read them.
    offsets: dict[int, int] = {}
    sizes: dict[int, int] = {}
    cursor = 0
    for det_id in det_ids:
        sizes[det_id] = shapes[det_id][0] * shapes[det_id][1] * itemsize
        offsets[det_id] = cursor
        cursor += sizes[det_id]
    block_size = cursor

    # Before the block is written, for the reason the push path checks too: an
    # undersized /dev/shm lets shm_open and ftruncate both succeed and then
    # kills the process with SIGBUS on the first write it cannot back.
    free = protocol.shm_free_bytes()
    if free is not None and free < block_size:
        raise ParallelEncodeUnavailable(
            f"{protocol.SHM_MOUNT} has {free / 1024 / 1024:.0f} MB free but this "
            f"exposure needs {block_size / 1024 / 1024:.0f} MB"
        )

    try:
        block = shared_memory.SharedMemory(create=True, size=block_size)
    except OSError as exc:
        raise ParallelEncodeUnavailable(f"cannot create shared block: {exc}") from exc

    view = None
    exposures: dict[str, Any] = {}
    try:
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT.update(
            shm_name=block.name,
            handles=dict(source.handles),
            slots={
                det_id: (offsets[det_id], sizes[det_id], shapes[det_id])
                for det_id in det_ids
            },
        )
        _fan_out(det_ids, n_workers, worker=_assemble_one)

        view = memoryview(block.buf)
        for det_id in det_ids:
            handle = source.handles[det_id]
            detector = cameraGeom.makeUpdatedDetector(camera[handle.detector_num])
            bbox = cameraGeomUtils.calcRawCcdBBox(detector)
            stub = _build_stub(handle, bbox, detector)

            array = np.ndarray(
                (bbox.getHeight(), bbox.getWidth()),
                dtype=exposure_codec.IMG_DTYPE,
                buffer=view,
                offset=offsets[det_id],
            )
            image = afwImage.ImageF(array=array, deep=False)
            image.setXY0(bbox.getMin())
            exposure = afwImage.ExposureF(afwImage.MaskedImageF(image))

            # Mask and variance come back all-zero, as they are on a raw and as
            # the push path's decode leaves them.
            exposure.setDetector(stub.getDetector())
            exposure.setWcs(stub.getWcs())
            exposure.setFilter(stub.getFilter())
            exposure.setMetadata(stub.getMetadata())
            exposure.getInfo().setVisitInfo(stub.getInfo().getVisitInfo())
            exposure.getInfo().setId(stub.getInfo().getId())
            exposures[detector.getName()] = exposure

        yield exposures
    finally:
        _WORKER_CONTEXT.clear()
        # Order is load-bearing: the exposures alias the block, so they go
        # first, then the view they were cut from, and only then can the block
        # be closed. `close()` raises BufferError while anything is exported.
        exposures.clear()
        if view is not None:
            view.release()
        block.close()
        try:
            block.unlink()
        except FileNotFoundError:
            pass


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
        # This path, and only this path, can encode in parallel: `load_raw` is
        # ours, so the work is known to be fork-safe and the image size is
        # known from cameraGeom before any pixel is read. The butler and
        # assembled-file sources keep the serial encoder.
        encoded_blob=encoded_blob,
    )
    return source, "/".join(sorted(image_types))
