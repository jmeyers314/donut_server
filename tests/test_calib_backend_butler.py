"""The butler calib backend: resolution semantics and the identity-keyed caches.

Needs a real butler repo, so the whole module skips when
DONUT_SERVER_BUTLER_REPO is unset -- the same convention test_refcat_store.py
uses for its data directories. The files backend is covered by
test_prepared_cache.py, which pins itself to `files` for the same reason this
module pins itself to `butler`.

What is worth testing here is *which* datasets get resolved rather than that
FITS parsing works: the loading code is shared with the files path, while the
resolution rule -- defer to the butler, do not filter the collection chain -- is
this backend's whole reason to exist and the thing a later refactor would most
plausibly "tidy up" into incorrectness.
"""
import os

import pytest

from lsst.ts.donut_server import coordinator

REPO = os.environ.get("DONUT_SERVER_BUTLER_REPO", "")
NO_REPO = f"needs DONUT_SERVER_BUTLER_REPO, got {REPO!r}"

pytestmark = pytest.mark.skipif(not REPO, reason=NO_REPO)

BORESIGHT = (283.666, -28.1326)

# The r_57 exposure's midpoint, which is what the client sends for it.
CALIB_TIME = "2026-07-14T05:42:02.206"

# Deliberately long before anything in this repo was certified. The point of
# this one is that it still resolves -- see the RUN-fallback test.
UNCERTIFIED_TIME = "2020-01-01T00:00:00"

N_DETECTORS = 8


def prepare_command(physical_filter: str = "r_57", calib_time: str = CALIB_TIME) -> dict:
    return {
        "physical_filter": physical_filter,
        "boresight_ra": BORESIGHT[0],
        "boresight_dec": BORESIGHT[1],
        "calib_time": calib_time,
        "config_overrides": [],
    }


@pytest.fixture(autouse=True)
def butler_backend(monkeypatch):
    """Pin the butler backend and reset both calib caches.

    The detector half is a slot that nothing evicts in production, so without
    this reset the first test to run leaves it populated and every later
    `detector_reused is False` assertion becomes order-dependent.
    """
    monkeypatch.setenv("DONUT_SERVER_CALIB_BACKEND", "butler")
    coordinator._FILTER_CALIB_CACHE.clear()
    coordinator._DETECTOR_CALIB_CACHE = None
    yield
    coordinator._FILTER_CALIB_CACHE.clear()
    coordinator._DETECTOR_CALIB_CACHE = None


def test_a_cold_resolve_finds_every_required_calib():
    calib, timings = coordinator._ensure_calib(prepare_command())

    assert timings["backend"] == "butler"
    assert timings["reused"] is False
    assert sorted(calib.ptc_by_name) == sorted(coordinator.CORNER_DETECTOR_NAMES.values())
    assert len(calib.linearizer_by_name) == N_DETECTORS
    assert len(calib.crosstalk_by_name) == N_DETECTORS
    assert len(calib.flat_by_name) == N_DETECTORS
    assert calib.physical_filter == "r_57"


def test_the_cache_keys_are_resolved_dataset_ids():
    """Not the physical_filter, which is the files backend's identity: a
    recertification changes which datasets a filter resolves to without changing
    the filter, and keying on the filter would then serve stale calibs forever.
    """
    ids = coordinator.calib_backend().resolve(prepare_command())

    # 3 detector-dimensioned types x 8, and 2 filter-dimensioned types x 8.
    assert len(ids.detector) == 3 * N_DETECTORS
    assert len(ids.filtered) == 2 * N_DETECTORS
    assert all(isinstance(i, type(next(iter(ids.detector)))) for i in ids.detector)
    # The two halves must not overlap, or an eviction of one would corrupt the
    # other's identity.
    assert not (ids.detector & ids.filtered)


def test_repeating_one_prepare_reuses_both_halves():
    coordinator._ensure_calib(prepare_command())
    _, timings = coordinator._ensure_calib(prepare_command())

    assert timings["reused"] is True
    assert timings["detector_reused"] is True
    assert timings["filter_reused"] is True
    assert timings["elapsed_s"] == 0.0
    # The resolve still happened -- it is what proves the cache is still valid.
    assert timings["resolve_s"] > 0.0


def test_a_filter_change_reloads_only_the_filter_dependent_half():
    """The same invariant the files backend has, and the reason the cache is
    split: the ptc/linearizer/crosstalk datasets are dimensioned by detector
    alone, so they resolve identically across filters and must not be re-read."""
    coordinator._ensure_calib(prepare_command("r_57"))
    detector = coordinator._DETECTOR_CALIB_CACHE[1]

    calib, timings = coordinator._ensure_calib(prepare_command("g_6"))

    assert timings["detector_reused"] is True
    assert timings["filter_reused"] is False
    assert calib.physical_filter == "g_6"
    assert calib.detector is detector


def test_an_uncertified_era_falls_back_to_the_producer_run():
    """The decision this backend turns on, pinned.

    pipetask passes its input collections unfiltered, RUN rows get a synthetic
    unbounded validity range, and find-first ranks by position in the chain with
    no preference for CALIBRATION. So at a time nothing was certified for, the
    answer comes from the producer run rather than being a miss -- and that is
    the answer the offline pipeline would have used, which is the only one this
    service is allowed to give.

    If this ever starts raising or returning nothing, someone has "fixed" the
    backend by filtering the chain to its CALIBRATION children. In this repo
    that makes a 2020 lookup resolve *nothing at all*.
    """
    ids = coordinator.calib_backend().resolve(prepare_command(calib_time=UNCERTIFIED_TIME))

    assert len(ids.detector) == 3 * N_DETECTORS
    assert len(ids.filtered) == 2 * N_DETECTORS


def test_a_resolve_leaves_no_butler_behind():
    """Fork safety. The task forks eight cutout workers per push, and an
    inherited live sqlite connection is unsupported even where it appears to
    work -- so the Butler is per-resolve and closed, never cached.

    Asserted rather than left to the `finally` being right, because the failure
    mode is not an error: it is undefined behaviour in a forked child, which
    looks like a flaky job.
    """
    backend = coordinator.calib_backend()
    calib, _ = coordinator._ensure_calib(prepare_command())

    from lsst.daf.butler import Butler

    assert not any(isinstance(v, Butler) for v in vars(backend).values())
    assert not any(isinstance(v, Butler) for v in vars(calib.detector).values())
    assert not any(isinstance(v, Butler) for v in vars(calib.filtered).values())


def test_a_malformed_calib_time_is_a_tagged_error():
    """So the front-end can answer 400 rather than 500: it validates the shape
    of this field but cannot parse it, having no astropy."""
    with pytest.raises(coordinator.CalibTimeError, match="not a parseable time"):
        coordinator._calib_timespan("halfway through tuesday")


def test_an_unknown_backend_names_the_valid_values(monkeypatch):
    monkeypatch.setenv("DONUT_SERVER_CALIB_BACKEND", "bogus")
    with pytest.raises(RuntimeError, match="'files'.*'butler'"):
        coordinator.calib_backend()


def test_the_butler_backend_requires_a_repo(monkeypatch):
    monkeypatch.delenv("DONUT_SERVER_BUTLER_REPO", raising=False)
    with pytest.raises(RuntimeError, match="DONUT_SERVER_BUTLER_REPO"):
        coordinator.calib_backend().resolve(prepare_command())
