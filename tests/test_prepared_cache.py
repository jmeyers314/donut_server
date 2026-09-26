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

# Same pointing throughout: what varies between commands below is the band,
# which is enough to force a distinct calib_key (and hence a distinct
# composite key) without needing four different boresights.
BORESIGHT = (283.666, -28.1326)
BANDS = ("r", "g", "i", "u")


def prepare_command(band: str) -> dict:
    return {
        "band": band,
        "boresight_ra": BORESIGHT[0],
        "boresight_dec": BORESIGHT[1],
        "config_overrides": [],
    }


@pytest.fixture(autouse=True)
def clean_cache():
    coordinator._PREPARED_CACHE.clear()
    coordinator._CALIB_CACHE.clear()
    coordinator._PREPARE_COMMANDS.clear()
    coordinator._ACTIVE_KEY = None
    yield
    coordinator._PREPARED_CACHE.clear()
    coordinator._CALIB_CACHE.clear()
    coordinator._PREPARE_COMMANDS.clear()
    coordinator._ACTIVE_KEY = None


def test_a_cold_prepare_builds_and_activates_an_entry():
    prepared = coordinator.ensure_prepared(prepare_command("r"))

    assert prepared["timings"]["task"]["reused"] is False
    assert prepared["timings"]["calib"]["reused"] is False
    assert coordinator._ACTIVE_KEY == prepared["key"]
    assert coordinator._CALIB_STORE["calib"].band == "r"


def test_repeating_the_same_prepare_hits_every_reuse_guard():
    first = coordinator.ensure_prepared(prepare_command("r"))
    second = coordinator.ensure_prepared(prepare_command("r"))

    assert second["key"] == first["key"]
    assert second["timings"]["task"]["reused"] is True
    assert second["timings"]["calib"]["reused"] is True


def test_prepare_b_then_push_a_reloads_a_rather_than_running_under_b():
    """prepare(A) -> prepare(B) -> push(A): A must still run under A's config,
    not silently under whatever B most recently loaded -- the exact hazard
    this cache exists to prevent.
    """
    a = coordinator.ensure_prepared(prepare_command("r"))
    coordinator.ensure_prepared(prepare_command("g"))
    assert coordinator._ACTIVE_KEY != a["key"]  # B is live now

    coordinator.ensure_prepared_for_push(a["key"])

    assert coordinator._ACTIVE_KEY == a["key"]
    assert coordinator._CALIB_STORE["calib"].band == "r"


def test_push_after_eviction_reloads_from_the_original_prepare_command():
    """Same as above, but A has actually been evicted from the cache (not just
    superseded as the active entry) by the time its push arrives.
    """
    a = coordinator.ensure_prepared(prepare_command("r"))
    coordinator.ensure_prepared(prepare_command("g"))
    coordinator.ensure_prepared(prepare_command("i"))
    coordinator.ensure_prepared(prepare_command("u"))  # cap=3 default: evicts A
    assert a["key"] not in coordinator._PREPARED_CACHE

    coordinator.ensure_prepared_for_push(a["key"])

    assert a["key"] in coordinator._PREPARED_CACHE
    assert coordinator._ACTIVE_KEY == a["key"]
    assert coordinator._CALIB_STORE["calib"].band == "r"


def test_a_pointing_change_at_one_band_reuses_the_calibs():
    """The reason _CALIB_CACHE exists: a slew big enough to change the level-5
    shard set is a new composite key, but the calibs depend only on the band and
    must survive it."""
    near = prepare_command("r")
    far = prepare_command("r") | {"boresight_ra": BORESIGHT[0] + 2.0}

    first = coordinator.ensure_prepared(near)
    calib = coordinator._CALIB_STORE["calib"]
    second = coordinator.ensure_prepared(far)

    assert second["key"] != first["key"]  # the pointing really did change the key
    assert second["timings"]["refcat"]["reused"] is False
    assert second["timings"]["calib"]["reused"] is True
    assert coordinator._CALIB_STORE["calib"] is calib


def test_calib_cap_evicts_by_band(monkeypatch):
    monkeypatch.setattr(coordinator, "CALIB_CACHE_CAP", 2)

    for band in ("r", "g", "i"):
        coordinator.ensure_prepared(prepare_command(band))

    assert "r" not in coordinator._CALIB_CACHE
    assert set(coordinator._CALIB_CACHE) == {"g", "i"}


def test_cap_evicts_least_recently_touched_first(monkeypatch):
    monkeypatch.setattr(coordinator, "PREPARED_CACHE_CAP", 3)

    keys = [coordinator.ensure_prepared(prepare_command(band))["key"] for band in BANDS]

    assert keys[0] not in coordinator._PREPARED_CACHE
    for key in keys[1:]:
        assert key in coordinator._PREPARED_CACHE


def test_a_push_touch_protects_an_entry_from_eviction(monkeypatch):
    """Pushing an entry counts as a touch, same as preparing it: a straggling
    push must not lose its config to an eviction caused by newer prepares."""
    monkeypatch.setattr(coordinator, "PREPARED_CACHE_CAP", 2)

    a = coordinator.ensure_prepared(prepare_command("r"))
    coordinator.ensure_prepared(prepare_command("g"))
    coordinator.ensure_prepared_for_push(a["key"])  # touch A -> MRU
    coordinator.ensure_prepared(prepare_command("i"))  # should evict B, not A

    assert a["key"] in coordinator._PREPARED_CACHE


def test_push_for_a_key_never_prepared_is_a_loud_error():
    bogus_key = ((), "r", frozenset({0}))
    with pytest.raises(RuntimeError, match="no prepared config"):
        coordinator.ensure_prepared_for_push(bogus_key)
