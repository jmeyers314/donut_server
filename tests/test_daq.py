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
# production entry points (`bin.src/donutTrigger.py`, and `daq.py` itself for
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
        f"{_UNCLAMPED}; the parallel encoder declines to fork here. Re-run with "
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


def _parts(blob):
    from lsst.ts.donut_server import protocol

    view = memoryview(blob)
    return {
        part.name: bytes(view[part.offset : part.offset + part.byte_count])
        for part in protocol.parse_layout(view)
    }


@pytest.fixture(scope="module")
def both_blobs(paired):
    """The same exposure encoded serially and in parallel."""
    from lsst.ts.donut_server import client, protocol

    source, _ = paired
    serial = protocol.pack_blob(client.build_raw_parts(source))
    with daq.encoded_blob(source) as view:
        return serial, bytes(view)


@needs_fork
def test_parallel_and_serial_agree_on_the_images(both_blobs):
    """The pixels are the whole payload, and they must be identical."""
    serial, parallel = both_blobs
    assert len(serial) == len(parallel)

    got, want = _parts(parallel), _parts(serial)
    assert sorted(got) == sorted(want)
    for name in sorted(want):
        if name.endswith(":img"):
            assert got[name] == want[name], name


@needs_fork
def test_parallel_and_serial_decode_to_the_same_exposures(both_blobs):
    """The meta parts, compared by what they rebuild rather than byte-wise.

    The two paths' meta pickles are *not* byte-identical, and deliberately so:
    the serial one slices its 1x1 stub out of a whole exposure so the stub
    carries a real pixel, while the parallel one builds the stub fresh and
    leaves it zero. `decode_exposure` never reads that pixel -- it fills every
    pixel from the image blob -- so this compares the decoded result, which is
    what actually crosses the wire.
    """
    from lsst.ts.donut_server import exposure_codec, protocol

    serial, parallel = both_blobs
    got, want = _parts(parallel), _parts(serial)

    for sensor, (img, meta) in protocol.split_parts(want).items():
        reference = exposure_codec.decode_exposure(img, meta)
        mine = exposure_codec.decode_exposure(
            got[protocol.part_name(sensor, protocol.IMG_SUFFIX)],
            got[protocol.part_name(sensor, protocol.META_SUFFIX)],
        )

        assert mine.getBBox() == reference.getBBox(), sensor
        assert np.array_equal(
            mine.getImage().getArray(), reference.getImage().getArray()
        ), sensor
        assert mine.getFilter() == reference.getFilter(), sensor
        assert mine.getWcs() == reference.getWcs(), sensor
        assert mine.getInfo().getVisitInfo() == reference.getInfo().getVisitInfo(), (
            sensor
        )
        assert mine.getInfo().getId() == reference.getInfo().getId(), sensor
        assert mine.getDetector().getName() == reference.getDetector().getName(), sensor
        assert [a.getRawBBox() for a in mine.getDetector()] == [
            a.getRawBBox() for a in reference.getDetector()
        ], sensor
        assert mine.getMetadata().toDict() == reference.getMetadata().toDict(), sensor


def test_serial_fallback_is_used_when_forced(paired, monkeypatch):
    """`DONUT_SERVER_SERIAL_ENCODE` is the A/B control and the escape hatch."""
    from lsst.ts.donut_server import client, protocol

    source, _ = paired
    monkeypatch.setenv(daq.SERIAL_ENCODE_VAR, "1")

    with pytest.raises(daq.ParallelEncodeUnavailable, match=daq.SERIAL_ENCODE_VAR):
        daq.build_parts_parallel(source)

    # And the manager degrades rather than propagating.
    with daq.encoded_blob(source) as view:
        assert bytes(view) == protocol.pack_blob(client.build_raw_parts(source))


@needs_fork
def test_a_failing_worker_names_its_detector(paired):
    """One unreadable sensor must not surface as a bare multiprocessing trace.

    Only the naming is asserted, not a fallback: an unreadable file fails the
    serial encoder too, and should. What matters is that the error says which
    detector, since a worker traceback carries no detector at all.
    """
    import dataclasses

    source, _ = paired
    broken = dict(source.handles)
    first = sorted(broken)[0]
    broken[first] = dataclasses.replace(broken[first], path="/nonexistent/raw.fits")
    bad = dataclasses.replace(source, handles=broken)

    with pytest.raises(daq.ParallelEncodeUnavailable, match=f"detector {first}"):
        daq.build_parts_parallel(bad)


@needs_fork
def test_a_worker_that_dies_without_reporting_times_out(paired, monkeypatch):
    """The failure mode that would otherwise wedge an exposure forever.

    `Pool.map` waits on a result queue, so a worker killed outright -- OOM
    killer, segfault in cfitsio, stray killpg -- never posts a result and the
    parent blocks indefinitely (verified: a SIGKILLed worker hangs `map` with
    no timeout). An *exception* in a worker is harmless by comparison, since
    `_encode_one` turns those into return values. Nothing supervises a detached
    trigger, so a hang here is an exposure that never finishes and never says
    why.
    """
    source, _ = paired
    monkeypatch.setattr(daq, "_encode_one", _suicidal_worker)
    monkeypatch.setattr(daq, "FAN_OUT_TIMEOUT_S", 8.0)

    with pytest.raises(daq.ParallelEncodeUnavailable, match="did not return"):
        daq.build_parts_parallel(source)


@needs_fork
def test_an_unwritable_block_falls_back_to_serial(paired, monkeypatch):
    """The real fallback case: the work is fine, the shared memory is not."""
    from lsst.ts.donut_server import client, protocol

    source, _ = paired

    def refuse(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(daq.shared_memory, "SharedMemory", refuse)

    with daq.encoded_blob(source) as view:
        assert bytes(view) == protocol.pack_blob(client.build_raw_parts(source))


@needs_fork
def test_a_small_shm_mount_falls_back_rather_than_risking_sigbus(paired, monkeypatch):
    """An undersized /dev/shm kills the process with SIGBUS on *write*, so the
    shortfall has to be caught before the block is written."""
    from lsst.ts.donut_server import protocol

    source, _ = paired
    monkeypatch.setattr(protocol, "shm_free_bytes", lambda *a, **k: 1024)

    with pytest.raises(daq.ParallelEncodeUnavailable, match="free"):
        daq.build_parts_parallel(source)


@needs_fork
def test_the_parallel_path_leaves_no_shared_blocks_behind(paired):
    """The creator unlinks; a leak per exposure would fill /dev/shm overnight."""
    import glob as _glob

    source, _ = paired
    before = set(_glob.glob("/dev/shm/psm_*"))
    with daq.encoded_blob(source) as view:
        assert len(view) > 0
    assert set(_glob.glob("/dev/shm/psm_*")) <= before


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
