"""The LRU cache of (task, calib, refcat) bundles that replaces the old
per-resource singleton globals -- see HANDOFF_prepare_lru_cache.md.

Needs real calib and refcat data (small: 8 detectors' worth), so it skips
whole when DONUT_SERVER_CALIB_DIR / DONUT_SERVER_REFCAT_DIR are unset, the
same convention test_refcat_store.py uses.
"""
import os

import pytest

from lsst.ts.donut_server import coordinator

CALIB_DIR = os.environ.get("DONUT_SERVER_CALIB_DIR", "")
REFCAT_DIR = os.environ.get("DONUT_SERVER_REFCAT_DIR", "")
NO_DATA = f"needs DONUT_SERVER_CALIB_DIR and DONUT_SERVER_REFCAT_DIR, got {CALIB_DIR!r}/{REFCAT_DIR!r}"

pytestmark = pytest.mark.skipif(not (CALIB_DIR and REFCAT_DIR), reason=NO_DATA)

# Same pointing throughout: what varies between commands below is the
# physical_filter, which is enough to force a distinct calib_key (and hence a
# distinct composite key) without needing four different boresights.
BORESIGHT = (283.666, -28.1326)
PHYSICAL_FILTERS = ("r_57", "g_6", "i_39", "u_24")

# The r_57 exposure's own midpoint. Held fixed across these commands so the
# physical_filter stays the only thing that varies; the files backend ignores it
# anyway, but it is part of the key, so a varying one would make every prepare
# below a distinct entry for a reason unrelated to what each test is checking.
CALIB_TIME = "2026-07-14T05:42:02.206"


def prepare_command(physical_filter: str) -> dict:
    return {
        "physical_filter": physical_filter,
        "boresight_ra": BORESIGHT[0],
        "boresight_dec": BORESIGHT[1],
        "calib_time": CALIB_TIME,
        "config_overrides": [],
    }


@pytest.fixture(autouse=True)
def files_backend(monkeypatch):
    """Pin the files backend: these tests describe the cache, not the backend,
    and a developer shell with DONUT_SERVER_CALIB_BACKEND=butler exported would
    otherwise silently retarget them at a repo."""
    monkeypatch.setenv("DONUT_SERVER_CALIB_BACKEND", "files")


@pytest.fixture(autouse=True)
def clean_cache():
    # _DETECTOR_CALIB_CACHE is never evicted in production, so it has to be
    # reset here or the first test to run leaves it populated and every later
    # `detector_reused is False` assertion becomes order-dependent.
    coordinator._PREPARED_CACHE.clear()
    coordinator._FILTER_CALIB_CACHE.clear()
    coordinator._DETECTOR_CALIB_CACHE = None
    coordinator._PREPARE_COMMANDS.clear()
    coordinator._ACTIVE_KEY = None
    yield
    coordinator._PREPARED_CACHE.clear()
    coordinator._FILTER_CALIB_CACHE.clear()
    coordinator._DETECTOR_CALIB_CACHE = None
    coordinator._PREPARE_COMMANDS.clear()
    coordinator._ACTIVE_KEY = None


def test_a_cold_prepare_builds_and_activates_an_entry():
    prepared = coordinator.ensure_prepared(prepare_command("r_57"))

    assert prepared["timings"]["task"]["reused"] is False
    assert prepared["timings"]["calib"]["reused"] is False
    assert coordinator._ACTIVE_KEY == prepared["key"]
    assert coordinator._CALIB_STORE["calib"].physical_filter == "r_57"


def test_repeating_the_same_prepare_hits_every_reuse_guard():
    first = coordinator.ensure_prepared(prepare_command("r_57"))
    second = coordinator.ensure_prepared(prepare_command("r_57"))

    assert second["key"] == first["key"]
    assert second["timings"]["task"]["reused"] is True
    assert second["timings"]["calib"]["reused"] is True


def test_prepare_b_then_push_a_reloads_a_rather_than_running_under_b():
    """prepare(A) -> prepare(B) -> push(A): A must still run under A's config,
    not silently under whatever B most recently loaded -- the exact hazard
    this cache exists to prevent.
    """
    a = coordinator.ensure_prepared(prepare_command("r_57"))
    coordinator.ensure_prepared(prepare_command("g_6"))
    assert coordinator._ACTIVE_KEY != a["key"]  # B is live now

    coordinator.ensure_prepared_for_push(a["key"])

    assert coordinator._ACTIVE_KEY == a["key"]
    assert coordinator._CALIB_STORE["calib"].physical_filter == "r_57"


def test_push_after_eviction_reloads_from_the_original_prepare_command():
    """Same as above, but A has actually been evicted from the cache (not just
    superseded as the active entry) by the time its push arrives.
    """
    a = coordinator.ensure_prepared(prepare_command("r_57"))
    coordinator.ensure_prepared(prepare_command("g_6"))
    coordinator.ensure_prepared(prepare_command("i_39"))
    coordinator.ensure_prepared(prepare_command("u_24"))  # cap=3 default: evicts A
    assert a["key"] not in coordinator._PREPARED_CACHE

    coordinator.ensure_prepared_for_push(a["key"])

    assert a["key"] in coordinator._PREPARED_CACHE
    assert coordinator._ACTIVE_KEY == a["key"]
    assert coordinator._CALIB_STORE["calib"].physical_filter == "r_57"


def test_a_pointing_change_at_one_filter_reuses_the_calibs():
    """The reason the calibs are cached apart from _PREPARED_CACHE: a slew big
    enough to change the level-5 shard set is a new composite key, but the
    calibs depend only on the physical_filter and must survive it."""
    near = prepare_command("r_57")
    far = prepare_command("r_57") | {"boresight_ra": BORESIGHT[0] + 2.0}

    first = coordinator.ensure_prepared(near)
    calib = coordinator._CALIB_STORE["calib"]
    second = coordinator.ensure_prepared(far)

    assert second["key"] != first["key"]  # the pointing really did change the key
    assert second["timings"]["refcat"]["reused"] is False
    assert second["timings"]["calib"]["reused"] is True
    # Identity of the two halves, not of the CalibSet: that is a view, composed
    # fresh per activation. Nothing being reloaded is the invariant here.
    now = coordinator._CALIB_STORE["calib"]
    assert now.detector is calib.detector
    assert now.filtered is calib.filtered


def test_a_filter_change_reloads_only_the_filter_dependent_half():
    """Why the calib cache is split in two: the PTCs, linearizers and crosstalk
    are dimensioned by detector alone, so re-reading them on a filter change was
    most of the rebuild time spent on bytes that had not changed."""
    first = coordinator.ensure_prepared(prepare_command("r_57"))
    detector = coordinator._CALIB_STORE["calib"].detector
    filtered = coordinator._CALIB_STORE["calib"].filtered

    second = coordinator.ensure_prepared(prepare_command("g_6"))

    assert second["key"] != first["key"]
    assert second["timings"]["calib"]["detector_reused"] is True
    assert second["timings"]["calib"]["filter_reused"] is False
    assert second["timings"]["calib"]["reused"] is False
    assert coordinator._CALIB_STORE["calib"].detector is detector
    assert coordinator._CALIB_STORE["calib"].filtered is not filtered


def test_calib_cap_evicts_by_physical_filter(monkeypatch):
    monkeypatch.setattr(coordinator, "CALIB_CACHE_CAP", 2)

    for physical_filter in ("r_57", "g_6", "i_39"):
        coordinator.ensure_prepared(prepare_command(physical_filter))

    assert "r_57" not in coordinator._FILTER_CALIB_CACHE
    assert set(coordinator._FILTER_CALIB_CACHE) == {"g_6", "i_39"}
    # The cap bounds only the filter-dependent half; the other one outlives
    # every eviction that happened above.
    assert coordinator._DETECTOR_CALIB_CACHE is not None


def test_cap_evicts_least_recently_touched_first(monkeypatch):
    monkeypatch.setattr(coordinator, "PREPARED_CACHE_CAP", 3)

    keys = [
        coordinator.ensure_prepared(prepare_command(pf))["key"]
        for pf in PHYSICAL_FILTERS
    ]

    assert keys[0] not in coordinator._PREPARED_CACHE
    for key in keys[1:]:
        assert key in coordinator._PREPARED_CACHE


def test_a_push_touch_protects_an_entry_from_eviction(monkeypatch):
    """Pushing an entry counts as a touch, same as preparing it: a straggling
    push must not lose its config to an eviction caused by newer prepares."""
    monkeypatch.setattr(coordinator, "PREPARED_CACHE_CAP", 2)

    a = coordinator.ensure_prepared(prepare_command("r_57"))
    coordinator.ensure_prepared(prepare_command("g_6"))
    coordinator.ensure_prepared_for_push(a["key"])  # touch A -> MRU
    coordinator.ensure_prepared(prepare_command("i_39"))  # should evict B, not A

    assert a["key"] in coordinator._PREPARED_CACHE


def test_push_for_a_key_never_prepared_is_a_loud_error():
    bogus_key = coordinator.PrepareKey((), "r_57", frozenset({0}), CALIB_TIME)
    with pytest.raises(RuntimeError, match="no prepared config"):
        coordinator.ensure_prepared_for_push(bogus_key)
