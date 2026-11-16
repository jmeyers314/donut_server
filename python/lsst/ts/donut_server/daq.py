"""Turn a directory of DAQ-native LSSTCam corner-sensor files into a `RawSource`.

The camera writes *unassembled amplifier* images -- one FITS per corner sensor
with eight `Segment1x` HDUs -- so something has to assemble them into the
whole-CCD `ExposureF` the task expects. That is this module, in-process, with
`lsst.obs.lsst`.

This is a distinct path, not a stopgap for one. The alternative shapes are a
producer that ships assembled exposures (`client.resolve_from_files`) and a
butler that has already ingested them (`client.resolve_from_butler`, i.e.
`donutClient.py -b`); all three exist, and this is the one the camera's data
arrives in. Which is why it is the one that gets the fast machinery: parallel
assembly, and `ingested_exposures` for the server-side path where there is no
wire format at all.

One caller: the coordinator's `/submit` and `/ingest` read the directory
in-process, assemble the pixels into shared memory and hand the task the
exposures directly, so no wire format is involved at any point. Both the
camera's hook and `scripts/run_sequence.py` reach it that way, by sending a
path rather than pixels, so neither of them imports this module.

Layout it expects, one directory per exposure, as the replay tree is laid out:

    <dir>/MC_O_20260713_000050_R00_SW0.fits
         /MC_O_20260713_000050_R00_SW0.json          # not read
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
with. See `replay` for why that matters and where the map comes from.

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
# single-threaded, because `ingested_exposures` forks. This is for fork
# safety, not speed -- A/B'd on the serial assembly loop at 1.92 s either way --
# but this numpy links OpenBLAS built with USE_OPENMP, and forking with live
# OpenMP threads is undefined behaviour.
#
# It has to be *here*, above the `client` import: that reaches
# `exposure_codec` and so afw and numpy, and these only bind if they are set
# before numpy loads. `bin.src/donutServer.py` sets the same five for the
# service, which is the only deployment that forks here; `setdefault` makes the
# overlap harmless, and this copy covers any other in-process importer.
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
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

from lsst.daf.base import PropertyList

from lsst.ts.donut_server import client
from lsst.ts.donut_server import exposure_codec
from lsst.ts.donut_server import tmpfs

if TYPE_CHECKING:
    from astro_metadata_translator import ObservationInfo

# The eight corner wavefront sensors, as DAQ filenames spell them. SW0 is
# extra-focal, SW1 intra-focal.
SW_PATTERN = "*_R??_SW?.fits"

EXPECTED_SENSORS = 8

# How long to wait for the eight workers before concluding one died without
# reporting. Deliberately generous against how long they actually take --
# this is the backstop for a worker that is *gone*, not a cadence control.
# See `_fan_out`.
FAN_OUT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class RawHandle:
    """One sensor's file, plus the header to assemble it under.

    A `client.RawSource.handles` value.

    Attributes
    ----------
    path : str
        The DAQ-native file the pixels are read from.
    header : PropertyList
        The primary header to assemble under, which may be a substitute that
        is not on disk anywhere near `path`. Carried rather than re-read by
        `load_raw`.
    detector_num : int
        Which detector this is.
    """

    path: str
    header: PropertyList
    detector_num: int


def read_header(path: str) -> PropertyList:
    """The primary header, with `astro_metadata_translator`'s fixes applied.

    Parameters
    ----------
    path : str
        A DAQ-native corner-sensor file.

    Returns
    -------
    PropertyList
        HDU 0's header, fixed in place.
    """
    from astro_metadata_translator import fix_header
    from lsst.afw.fits import readMetadata
    from lsst.obs.lsst.translators import LsstCamTranslator

    metadata = readMetadata(path, 0)
    fix_header(metadata, translator_class=LsstCamTranslator)
    return metadata


def obs_info(
    header: PropertyList, filename: str | None = None
) -> ObservationInfo:
    """`ObservationInfo` for one sensor, from its primary header alone.

    Parameters
    ----------
    header : PropertyList
        A fixed primary header, live or substituted.
    filename : str, optional
        Reported in the translator's own error messages, which otherwise
        cannot say which sensor was being read.

    Returns
    -------
    ObservationInfo
        The translated view of `header`.

    Notes
    -----
    Deliberately header-only: this is what makes the /prepare half of the cycle
    free of pixel reads. Eight of these cost ~0.1 s total, against ~0.1 s per
    sensor to assemble.
    """
    from astro_metadata_translator import ObservationInfo
    from lsst.obs.lsst.translators import LsstCamTranslator

    return ObservationInfo(
        header, translator_class=LsstCamTranslator, filename=filename
    )


def read_obs_info(path: str) -> ObservationInfo:
    """`ObservationInfo` straight off a file, for callers with no substitute.

    Parameters
    ----------
    path : str
        A DAQ-native corner-sensor file.

    Returns
    -------
    ObservationInfo
        The translated view of that file's primary header.
    """
    return obs_info(read_header(path), filename=path)


def detector_name(header: PropertyList) -> str:
    """`R00_SW0`-style name, as the coordinator's calib dicts are keyed.

    Parameters
    ----------
    header : PropertyList
        A primary header carrying `RAFTBAY` and `CCDSLOT`.

    Returns
    -------
    str
        `<RAFTBAY>_<CCDSLOT>`.

    Raises
    ------
    RuntimeError
        If either card is absent.
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
# translator discovery it provokes is not detector-specific -- so this is just
# the first of the eight.
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
            "bin.src/donutServer.py)."
        )


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



def _assemble_one(det_id: int) -> tuple[int, str | None]:
    """Write one detector's pixels into the shared block. Returns no pixels.

    Runs in a fork worker. The assembled image -- 37.75 MB -- is written
    *through* a view onto the parent's block, so it is never materialized in
    this child's heap and never pickled back. No metadata comes back either:
    the parent holds the same header and rebuilding a stub costs 0.0135 s
    against 0.037 s merely to unpickle one.

    Returns `(det_id, error)`, with `error` None on success. Errors come back
    as a value rather than an exception so that one unreadable sensor names
    itself, instead of surfacing as a `multiprocessing` traceback with no
    detector in it.
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

        return det_id, None
    except Exception as exc:  # noqa: BLE001 -- reported, not raised; see docstring
        return det_id, f"{type(exc).__name__}: {exc}"


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



def _fan_out(
    det_ids: list[int],
    n_workers: int | None,
    timeout: float | None = None,
) -> None:
    """Assemble `det_ids` into the shared block, one fork worker per detector.

    Returns nothing: the pixels are already in the block the parent set up, and
    the only thing to report back is failure. Raises
    `ParallelEncodeUnavailable` naming every detector that failed.

    `fork`, not `spawn`: the whole point is that the children inherit the camera
    and the loaded stack. That is safe here and not in the front end because
    this process is single-threaded by the time it gets here -- which is what
    the env block at the top of this module and `_assert_single_threaded_blas`
    together guarantee.

    `timeout` is not belt-and-braces: `Pool.map` waits on a result queue, so a
    worker that dies *without returning* -- SIGKILL from the OOM killer, a
    segfault in cfitsio, a stray killpg -- leaves it blocked forever (verified:
    a SIGKILLed worker hangs `map` indefinitely). An exception in a worker is
    fine, and `_assemble_one` converts those to return values anyway; it is
    only death that strands the parent. The coordinator's command loop is serial, so
    a parent blocked here does not merely lose this exposure: it stalls every
    later command on that flight, with nothing to say why.
    """
    # Read at call time, not bound as a default, so the module global stays the
    # one place to change it (and tests can shorten it).
    if timeout is None:
        timeout = FAN_OUT_TIMEOUT_S

    workers = min(n_workers or len(det_ids), len(det_ids))
    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers)
    try:
        results = pool.map_async(_assemble_one, det_ids).get(timeout=timeout)
    except mp.TimeoutError as exc:
        raise ParallelEncodeUnavailable(
            f"a worker did not return within {timeout:g}s; assuming one died "
            "without reporting (OOM killer, segfault, or external kill)"
        ) from exc
    except Exception as exc:
        raise ParallelEncodeUnavailable(f"fork pool failed: {exc}") from exc
    finally:
        # terminate() before join(), always: a bare join() waits for children
        # to finish on their own, which is exactly what the timeout above has
        # just established will not happen. This is what reaps the survivors,
        # and a child left running would hold a core in the coordinator for the
        # life of the process.
        pool.terminate()
        pool.join()

    failures = [
        f"detector {det_id}: {error}" for det_id, error in results if error is not None
    ]
    if failures:
        raise ParallelEncodeUnavailable("; ".join(failures))


@contextlib.contextmanager
def ingested_exposures(source, n_workers: int | None = None):
    """`{detector_name: ExposureF}` for `source`, assembled in this process.

    How the coordinator gets pixels. The consumer is in this process, so there
    is no wire format, no pickled pixels and no decode step at all.

    Three things make it cheap, all measured:

    - Eight forked workers assemble straight into one shared block, one per
      corner sensor.
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

    There is no serial fallback. The caller is the coordinator, which has no
    second way to get pixels, so a failure here is a failed job --
    `ParallelEncodeUnavailable` propagates and the command replies with it.
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

    tmpfs.require_free(block_size, "this exposure", ParallelEncodeUnavailable)

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
        _fan_out(det_ids, n_workers)

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

    Parameters
    ----------
    directory : str
        One exposure's directory.

    Returns
    -------
    list of str
        Every file matching `SW_PATTERN`, joined onto `directory` as it was
        given.

    Raises
    ------
    RuntimeError
        If anything other than exactly `EXPECTED_SENSORS` paths are found.
    """
    paths = sorted(glob.glob(os.path.join(directory, SW_PATTERN)))
    if len(paths) != EXPECTED_SENSORS:
        raise RuntimeError(
            f"found {len(paths)} corner-sensor files in {directory!r}, "
            f"expected {EXPECTED_SENSORS}"
        )
    return paths


def resolve_exposure_dir(
    directory: str, override_headers: dict[str, PropertyList] | None = None
) -> tuple[client.RawSource, str]:
    """Build a `RawSource` for one exposure directory, reading no pixels.

    Parameters
    ----------
    directory : str
        One exposure's directory. Must hold exactly `EXPECTED_SENSORS`
        corner-sensor files.
    override_headers : dict, optional
        Detector name -> primary header.  If present, overrides the headers
        from the files themselves. Useful for camera replay mode, which
        replays the original pixels but not the original headers.

    Returns
    -------
    source : client.RawSource
        Raw handles + meta for exposure.
    image_type : str
        The exposure's `IMGTYPE`, lowercased by the translator.

    Raises
    ------
    RuntimeError
        If the sensors disagree about anything that describes the exposure as
        a whole, if any of them cannot be identified, if there are not exactly
        `EXPECTED_SENSORS` of them, or if `override_headers` is incomplete.

    Notes
    -----
    Populate pointing and calib time from headers here to facilitate potential
    early returns later.
    """
    paths = sw_paths(directory)

    handles: dict[int, RawHandle] = {}
    filters: set[str] = set()
    exposure_ids: set[int] = set()
    image_types: set[str] = set()
    pointings: set[tuple[float, float] | None] = set()
    midpoints: set[str] = set()

    for path in paths:
        live = read_header(path)
        name = detector_name(live)
        # The live header still decides which detector this is, and so which
        # amp layout the pixels are in; only the observation metadata can be
        # substituted. See the module docstring.
        detector_num = obs_info(live, filename=path).detector_num
        if detector_num is None:
            raise RuntimeError(
                f"{os.path.basename(path)} has no detector number, so its "
                "pixels cannot be matched to an amp layout"
            )

        header = live
        if override_headers:
            if name not in override_headers:
                raise RuntimeError(
                    f"no substitute header for detector {name} "
                    f"({os.path.basename(path)}); have "
                    f"{sorted(override_headers)}"
                )
            header = override_headers[name]

        info = obs_info(header, filename=path)
        handles[detector_num] = RawHandle(path, header, detector_num)
        filters.add(str(info.physical_filter))
        exposure_ids.add(int(info.exposure_id))
        image_types.add(str(info.observation_type))
        radec = info.tracking_radec
        pointings.add(
            None if radec is None else (radec.ra.deg, radec.dec.deg)
        )
        begin, end = info.datetime_begin, info.datetime_end
        midpoints.add((begin + (end - begin) / 2).tai.isot)

    # Ensure all detector metadata are consistent.
    if len(filters) != 1:
        raise RuntimeError(f"{directory!r} spans physical_filters {filters}")
    if len(exposure_ids) != 1:
        raise RuntimeError(f"{directory!r} spans exposures {exposure_ids}")
    if len(image_types) != 1:
        raise RuntimeError(f"{directory!r} spans image types {image_types}")
    if len(pointings) != 1:
        raise RuntimeError(f"{directory!r} spans pointings {pointings}")
    if len(midpoints) != 1:
        raise RuntimeError(f"{directory!r} spans midpoints {midpoints}")
    if len(handles) != EXPECTED_SENSORS:
        raise RuntimeError(
            f"{directory!r} has {len(handles)} corner sensor(s), expected "
            f"{EXPECTED_SENSORS}: detectors {sorted(handles)}"
        )

    (physical_filter,) = filters
    (exposure_id,) = exposure_ids
    (image_type,) = image_types
    (boresight,) = pointings
    (calib_time,) = midpoints

    source = client.RawSource(
        visit=exposure_id,
        physical_filter=client._check_known_filter(exposure_id, physical_filter),
        handles=handles,
        load=load_raw,
        boresight=boresight,
        calib_time=calib_time,
    )
    return source, image_type
