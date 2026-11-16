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

# Whether this interpreter can fork safely, decided once at collection.
#
# The thread-limit env vars only bind if they are set before numpy loads, and
# under pytest that ship has already sailed: the `pytest_doctestplus` plugin
# imports numpy while plugins are being loaded, which is before any conftest or
# test module runs. So there is nowhere inside the test suite early enough to
# set them, and `daq` -- correctly -- declines to fork into a process with 128
# live OpenMP threads.
#
# Hence a skip rather than a workaround, with the fix in the reason. The
# production entry points (`bin.src/donutServer.py`, and `daq.py` itself for
# in-process callers) set these before anything reaches numpy, so this is a
# pytest-only wrinkle, not something a deployment has to think about.
_UNCLAMPED = None
try:
    daq._assert_single_threaded_blas()
except RuntimeError as exc:
    _UNCLAMPED = str(exc).split(".")[0]

needs_fork = pytest.mark.skipif(
    _UNCLAMPED is not None,
    reason=(
        f"{_UNCLAMPED}; the assembly fan-out declines to fork here. Re-run with "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 to exercise it"
    ),
)

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


def _suicidal_worker(det_id):
    """Die outright, the way an OOM kill or a segfault would.

    Module level, not a closure: `Pool.map` pickles the callable by qualified
    name, so a function defined inside a test fails to pickle and the pool
    reports *that* instead of the hang under test.
    """
    import signal

    os.kill(os.getpid(), signal.SIGKILL)


@needs_fork
def test_a_failing_worker_names_its_detector(paired):
    """One unreadable sensor must not surface as a bare multiprocessing trace.

    What matters is that the error says which detector: a worker traceback
    carries none, so without `_assemble_one` reporting the failure as a value
    an unreadable sensor would be a traceback with nothing to act on.
    """
    import dataclasses

    source, _ = paired
    broken = dict(source.handles)
    first = sorted(broken)[0]
    broken[first] = dataclasses.replace(broken[first], path="/nonexistent/raw.fits")
    bad = dataclasses.replace(source, handles=broken)

    with pytest.raises(daq.ParallelEncodeUnavailable, match=f"detector {first}"):
        with daq.ingested_exposures(bad):
            pass


@needs_fork
def test_a_worker_that_dies_without_reporting_times_out(paired, monkeypatch):
    """The failure mode that would otherwise wedge an exposure forever.

    `Pool.map` waits on a result queue, so a worker killed outright -- OOM
    killer, segfault in cfitsio, stray killpg -- never posts a result and the
    parent blocks indefinitely (verified: a SIGKILLed worker hangs `map` with
    no timeout). An *exception* in a worker is harmless by comparison, since
    `_assemble_one` turns those into return values. The coordinator's command
    loop is serial, so a hang here stalls every later command on that flight.
    """
    source, _ = paired
    monkeypatch.setattr(daq, "_assemble_one", _suicidal_worker)
    monkeypatch.setattr(daq, "FAN_OUT_TIMEOUT_S", 8.0)

    with pytest.raises(daq.ParallelEncodeUnavailable, match="did not return"):
        with daq.ingested_exposures(source):
            pass


@needs_fork
def test_a_small_shm_mount_is_refused_rather_than_risking_sigbus(paired, monkeypatch):
    """An undersized /dev/shm kills the process with SIGBUS on *write*, so the
    shortfall has to be caught before the block is written."""
    from lsst.ts.donut_server import tmpfs

    source, _ = paired
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: 1024)

    with pytest.raises(daq.ParallelEncodeUnavailable, match="free"):
        with daq.ingested_exposures(source):
            pass


@needs_fork
def test_ingested_exposures_match_load_raw(paired):
    """The server-side ingest path against the whole-exposure entry point.

    `ingested_exposures` assembles in forked workers, wraps each shared-memory
    slot zero-copy, and rebuilds the metadata in the parent rather than
    unpickling a stub from the child -- three places to get it wrong. This is
    the guard: the exposures it yields must be what `load_raw` would have
    produced, which `test_assembled_image_matches_the_reference` has already
    tied to an independent reference.
    """
    source, _ = paired
    reference = {
        daq.load_raw(source.handles[det]).getDetector().getName(): daq.load_raw(
            source.handles[det]
        )
        for det in sorted(source.handles)
    }

    with daq.ingested_exposures(source) as exposures:
        assert sorted(exposures) == sorted(reference)
        for name, got in sorted(exposures.items()):
            want = reference[name]
            assert np.array_equal(
                got.getImage().getArray(), want.getImage().getArray()
            ), f"{name} pixels"
            assert got.getBBox() == want.getBBox(), f"{name} bbox"
            assert got.getFilter() == want.getFilter(), f"{name} filter"
            assert got.getInfo().getId() == want.getInfo().getId(), f"{name} id"
            assert got.getWcs() == want.getWcs(), f"{name} wcs"
            assert [a.getRawBBox() for a in got.getDetector()] == [
                a.getRawBBox() for a in want.getDetector()
            ], f"{name} amps"
            assert (
                got.getMetadata().toDict() == want.getMetadata().toDict()
            ), f"{name} metadata"


@needs_fork
def test_ingested_exposures_alias_the_shared_block(paired):
    """Zero-copy is the point, so assert it rather than trusting it.

    A copy here would be ~300 MB and ~0.2 s per exposure, and nothing else in
    the result would look different -- which is exactly the kind of regression
    that survives a correctness test.

    Checked by *address*, not with `np.shares_memory`: afw hands each image's
    array back behind its own PyCapsule, so numpy cannot see that two of them
    have a common base and reports False even when they are windows onto one
    block. What it cannot hide is that the eight images land at exactly their
    slot offsets -- contiguous, one image apart -- which a per-exposure copy
    could not reproduce.
    """
    source, _ = paired
    with daq.ingested_exposures(source) as exposures:
        arrays = [e.getImage().getArray() for e in exposures.values()]
        addresses = sorted(a.__array_interface__["data"][0] for a in arrays)
        stride = arrays[0].size * arrays[0].dtype.itemsize
        gaps = {later - earlier for earlier, later in zip(addresses, addresses[1:])}
        assert gaps == {stride}, (
            f"images are {sorted(gaps)} bytes apart, expected one image "
            f"({stride}) -- they are not slots in a single block"
        )


@needs_fork
def test_ingest_releases_its_block(paired):
    """A leak per exposure would fill /dev/shm overnight, and the exposures
    alias the block -- so the release order inside the manager is load-bearing."""
    import glob as _glob

    source, _ = paired
    before = set(_glob.glob("/dev/shm/psm_*"))
    with daq.ingested_exposures(source) as exposures:
        assert len(exposures) == 8
    assert set(_glob.glob("/dev/shm/psm_*")) <= before


def test_warm_up_needs_no_exposure_and_leaves_logging_alone():
    """The coordinator calls this before its hello, so it must not need data.

    Driven by a synthetic header on purpose -- at server startup no exposure
    exists yet. It also suppresses the translator's complaints about that
    header's missing cards, so this checks the suppression is scoped: a
    `logging.disable` left in place would silence the whole coordinator.
    """
    import logging

    before = logging.root.manager.disable
    daq.warm_up()
    assert logging.root.manager.disable == before
    assert logging.getLogger("probe").isEnabledFor(logging.WARNING)


def test_warm_up_is_cheap_to_repeat():
    """Idempotent, so a second caller (or a retry) costs nothing."""
    import time

    daq.warm_up()
    t0 = time.monotonic()
    daq.warm_up()
    assert time.monotonic() - t0 < 0.1


def test_warm_up_makes_the_first_real_wcs_cheap(paired):
    """What the warm-up is actually for.

    `attachRawWcsFromBoresight`'s first call in a process spends ~0.5 s letting
    `astro_metadata_translator` work out which translator a header belongs to,
    and ~0.007 s on every call after. Without the warm-up that lands on the
    first exposure -- and in all eight fork workers at once.
    """
    import time

    import lsst.afw.image as afwImage
    import lsst.geom as geom
    from lsst.obs.lsst.assembly import attachRawWcsFromBoresight

    source, _ = paired
    handle = source.handles[sorted(source.handles)[0]]

    daq.warm_up()

    stub = afwImage.ExposureF(geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(1, 1)))
    stub.setDetector(daq._camera()[handle.detector_num])
    stub.setMetadata(handle.header)

    t0 = time.monotonic()
    attachRawWcsFromBoresight(stub)
    elapsed = time.monotonic() - t0

    # Generous against the ~0.007 s it actually takes once warm, but far below
    # the ~0.5 s it costs cold.
    assert elapsed < 0.15, f"first real WCS took {elapsed:.3f}s; warm-up did not take"


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
