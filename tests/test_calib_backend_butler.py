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
import functools
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


@functools.cache
def calibs_in_chain_runs() -> frozenset[str]:
    """Required calib types that a plain RUN in the configured chain carries.

    The RUN-fallback test below asserts a property of the *repo*, not of this
    code: that the collection chain includes the producer runs the calibs were
    built in. A repo built by scripts/build_blitz_repo.py does, because its
    discover_children chains every RUN it finds. A production repo's own
    LSSTCam/defaults does not -- there the chained RUNs are raws, refcats and
    skymaps, and the producer runs exist but are unchained -- so the fallback
    has nothing to fall back to and the test is inapplicable rather than
    failing.

    Cached because it opens a Butler, and closed before returning for the same
    fork-safety reason resolve() does it.
    """
    from lsst.daf.butler import Butler, MissingDatasetTypeError

    backend = coordinator.ButlerCalibBackend
    required = [
        t
        for t in backend.DETECTOR_TYPES + backend.FILTER_TYPES
        if t not in backend.OPTIONAL_TYPES
    ]
    butler = Butler.from_config(coordinator.butler_repo(), writeable=False)
    try:
        flat = coordinator._flatten_collections(butler, coordinator.butler_collections())
        runs = [
            info.name
            for info in butler.collections.query_info(flat)
            if info.type.name == "RUN"
        ]
        present = set()
        for dataset_type in required:
            try:
                # limit=1: an existence probe, not an inventory.
                if butler.query_datasets(
                    dataset_type,
                    collections=runs,
                    find_first=False,
                    explain=False,
                    limit=1,
                ):
                    present.add(dataset_type)
            except MissingDatasetTypeError:
                # Not registered at all, which is a stronger form of absent.
                continue
        return frozenset(present)
    finally:
        butler.close()


def test_an_uncertified_era_falls_back_to_the_producer_run():
    """The decision this backend turns on, pinned.

    pipetask passes its input collections unfiltered, RUN rows get a synthetic
    unbounded validity range, and find-first ranks by position in the chain with
    no preference for CALIBRATION. So at a time nothing was certified for, the
    answer comes from the producer run rather than being a miss -- and that is
    the answer the offline pipeline would have used, which is the only one this
    service is allowed to give.

    If this ever starts raising or returning nothing *on a repo that chains its
    producer runs*, someone has "fixed" the backend by filtering the chain to
    its CALIBRATION children: in such a repo that makes a 2020 lookup resolve
    nothing at all. The skip below is why that qualifier is there -- see
    calibs_in_chain_runs.
    """
    backend = coordinator.ButlerCalibBackend
    required = frozenset(
        t
        for t in backend.DETECTOR_TYPES + backend.FILTER_TYPES
        if t not in backend.OPTIONAL_TYPES
    )
    missing = required - calibs_in_chain_runs()
    if missing:
        pytest.skip(
            f"{coordinator.butler_repo()!r} chains no RUN carrying "
            f"{sorted(missing)}, so an uncertified lookup has no producer run "
            "to fall back to; needs a repo whose chain includes them"
        )

    ids = coordinator.calib_backend().resolve(prepare_command(calib_time=UNCERTIFIED_TIME))

    assert len(ids.detector) == 3 * N_DETECTORS
    assert len(ids.filtered) == 2 * N_DETECTORS


def test_a_resolve_leaves_no_butler_behind():
    """Fork safety. The task forks eight cutout workers per push, and an
    inherited live registry connection is unsupported even where it appears to
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


# ------------------------------------------------- the startup config check


def test_check_passes_on_the_real_repo():
    """The baseline for the three failure cases below: this repo is fine."""
    coordinator.check_calib_config()


def test_a_mistyped_repo_path_is_a_fatal_config_error(monkeypatch):
    """The bug this check exists for: `-b /Users/jmeyers/repo` for jmeyers3.

    Before it, the server came up healthy on a repo that does not exist -- both
    flights spawned, /health read ready -- and every /prepare then 500'd, with
    the only explanation in the HTTP response body.

    The typo is appended rather than substituted into REPO: a substitution only
    mangles the paths it matches, and silently passed once REPO stopped being
    one of them -- the suite ran against an alias (`/repo/main`) that the
    substitution left untouched, so this asserted that the *real* repo fails.
    """
    monkeypatch.setenv("DONUT_SERVER_BUTLER_REPO", REPO.rstrip("/") + "-mistyped")
    with pytest.raises(coordinator.CalibConfigError, match="not a usable butler repo"):
        coordinator.check_calib_config()


def test_collections_naming_nothing_are_a_fatal_config_error(monkeypatch):
    """The other way to configure a repo that can never resolve a calib: a real
    repo, and collections it does not contain.

    An unknown name raises out of query_info rather than reaching
    _flatten_collections' empty-result branch, so both paths are wrapped. That
    branch stays reachable for a chain that exists and is empty, which this
    read-only repo has no way to offer.
    """
    monkeypatch.setenv("DONUT_SERVER_BUTLER_COLLECTIONS", "LSSTCam/no-such-collection")
    with pytest.raises(coordinator.CalibConfigError, match="cannot be resolved"):
        coordinator.check_calib_config()


def test_an_unknown_backend_is_a_fatal_config_error(monkeypatch):
    """calib_backend()'s own RuntimeError, retagged: donutServer.py's banner
    warns that an unknown backend will fail, and this is what makes it fatal
    at startup rather than at the first /prepare."""
    monkeypatch.setenv("DONUT_SERVER_CALIB_BACKEND", "bogus")
    with pytest.raises(coordinator.CalibConfigError, match="'files'.*'butler'"):
        coordinator.check_calib_config()


def test_the_check_leaves_no_butler_behind():
    """Same fork-safety contract as a resolve: it runs before the hello, so a
    Butler stashed on the backend would be live for every later fork."""
    from lsst.daf.butler import Butler

    backend = coordinator.calib_backend()
    backend.check()
    assert not any(isinstance(v, Butler) for v in vars(backend).values())
