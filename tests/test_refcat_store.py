"""Guards on the level-5 -> level-7 reference catalog bridge.

The geometry and schema tests need no data. The reshard round trip does, and is
skipped when `ref_cat/` is absent, like the raw codec test.
"""
import glob
import os

import numpy as np
import pytest

from lsst.ts.donut_server import refcat_store
from lsst.meas.algorithms.loadReferenceObjects import getRefFluxField
from lsst.sphgeom import HtmPixelization, UnitVector3d

# Unset is a skip, not an error: these tests are data-dependent by design, and
# the directories are the user's to locate. os.environ directly rather than
# refcat_store.refcat_dir() / client.raw_dir(), which raise when unset.
SHARD_DIR = os.environ.get("DONUT_SERVER_REFCAT_DIR", "")
SHARD_PATHS = sorted(glob.glob(os.path.join(SHARD_DIR, "*.fits"))) if SHARD_DIR else []
NO_SHARDS = f"no shards in {SHARD_DIR or '$DONUT_SERVER_REFCAT_DIR (unset)'}"
RAW_DIR = os.environ.get("DONUT_SERVER_RAW_DIR", "")
RAW_PATHS = sorted(glob.glob(os.path.join(RAW_DIR, "raw_*_r.fits"))) if RAW_DIR else []
NO_RAWS = f"no raw_*.fits in {RAW_DIR or '$DONUT_SERVER_RAW_DIR (unset)'}"

# The r-band exposure's boresight, and the level-5 shards a FIELD_RADIUS_DEG
# circle about it covers. Recorded from a measurement, not derived, so that a
# change in FIELD_RADIUS_DEG or in the envelope logic shows up here.
BORESIGHT = (283.666, -28.1326)
EXPECTED_SHARDS = [11348, 11349, 11351, 11354, 11356, 11376, 11378, 11379, 11385, 11389]


def test_shard_ids_for_known_pointing():
    assert refcat_store.shard_ids_for_pointing(*BORESIGHT) == EXPECTED_SHARDS


def test_field_radius_reaches_past_the_corner_sensors():
    """The real check on FIELD_RADIUS_DEG: derive the reach from the raws.

    A literal comparison would not catch the thing that actually goes wrong here
    -- the margin the loader queries drifting away from COVERAGE_MARGIN_PX.
    """
    if not RAW_PATHS:
        pytest.skip(NO_RAWS)

    import lsst.afw.image as afwImage
    import lsst.geom as geom

    worst = 0.0
    for path in RAW_PATHS:
        exp = afwImage.ExposureF.readFits(path)
        boresight = exp.getInfo().getVisitInfo().boresightRaDec
        box = geom.Box2D(exp.getBBox())
        box.grow(refcat_store.COVERAGE_MARGIN_PX)
        wcs = exp.getWcs()
        for corner in box.getCorners():
            worst = max(worst, boresight.separation(wcs.pixelToSky(corner)).asDegrees())

    assert worst < refcat_store.FIELD_RADIUS_DEG, (
        f"corner sensors reach {worst:.4f} deg but FIELD_RADIUS_DEG is "
        f"{refcat_store.FIELD_RADIUS_DEG}"
    )


def test_coverage_margin_matches_the_loader():
    # loadPixelBox grows the bbox by the task's pixelMargin and then _makeBoxRegion
    # grows the outer region -- the one shard selection uses -- by
    # bboxToSpherePadding. Both terms must be in COVERAGE_MARGIN_PX.
    import inspect

    from lsst.meas.algorithms import ReferenceObjectLoader

    signature = inspect.signature(ReferenceObjectLoader.loadPixelBox)
    assert (
        signature.parameters["bboxToSpherePadding"].default
        == refcat_store.LOADER_BBOX_PADDING_PX
    )
    assert refcat_store.COVERAGE_MARGIN_PX == (
        refcat_store.TASK_PIXEL_MARGIN_PX + refcat_store.LOADER_BBOX_PADDING_PX
    )


def test_shard_ids_reject_bad_pointings():
    with pytest.raises(ValueError):
        refcat_store.shard_ids_for_pointing(float("nan"), 0.0)
    with pytest.raises(ValueError):
        refcat_store.shard_ids_for_pointing(0.0, 91.0)
    with pytest.raises(ValueError):
        refcat_store.shard_ids_for_pointing(*BORESIGHT, radius_deg=0.0)


def test_shard_ids_are_all_level_5():
    for shard_id in refcat_store.shard_ids_for_pointing(*BORESIGHT):
        assert HtmPixelization.level(shard_id) == refcat_store.SHARD_LEVEL


def test_load_level_matches_the_connection():
    # If upstream ever reshards the_monster to another HTM level, the connection
    # dimension moves and LOAD_LEVEL has to move with it. Without this the
    # mismatch would surface as an unbuildable DatasetRef deep inside a push.
    from lsst.ts.wep.blitz.donutBlitzCorner import (
        DonutBlitzCornerConfig,
        DonutBlitzCornerTask,
    )

    config = DonutBlitzCornerConfig()
    conns = config.connections.ConnectionsClass(config=config)
    (dimension,) = tuple(conns.refCat.dimensions)
    assert dimension == refcat_store.LOAD_DIMENSION


# A level-7 child of shard 12345, used for the dataId tests below.
HTM_INDEX = 197520


@pytest.fixture(scope="module")
def universe():
    from lsst.daf.butler import DimensionUniverse

    return DimensionUniverse()


def test_htm_data_id_carries_its_region(universe):
    data_id = refcat_store.htm_data_id(universe, HTM_INDEX)
    assert data_id.hasRecords()
    assert data_id.region == HtmPixelization(refcat_store.LOAD_LEVEL).pixel(HTM_INDEX)


def test_htm_data_id_region_is_absent_without_the_record(universe):
    # The failure this guards against: standardize() alone looks fine but leaves
    # the region unavailable, and the loader dereferences it.
    from lsst.daf.butler import DataCoordinate

    bare = DataCoordinate.standardize(
        {refcat_store.LOAD_DIMENSION: HTM_INDEX}, universe=universe
    )
    assert not bare.hasRecords()


def test_htm_data_id_keeps_the_index(universe):
    data_id = refcat_store.htm_data_id(universe, HTM_INDEX)
    assert data_id[refcat_store.LOAD_DIMENSION] == HTM_INDEX


def test_schema_is_a_single_shared_object():
    # loadRegion raises TypeError on any schema mismatch between the first shard
    # and the rest, so every catalog must be built against the same object.
    assert refcat_store.schema() is refcat_store.schema()


@pytest.mark.parametrize("filt", refcat_store.FLUX_FILTERS)
def test_flux_fields_resolve_and_are_nanojansky(filt):
    schema = refcat_store.schema()
    # nJy units are what keep the loader from falling through to
    # getFormatVersionFromRefCat(), which raises on these files: they carry no
    # REFCAT_FORMAT_VERSION metadata at all.
    assert schema.find(f"{filt}_flux").field.getUnits() == "nJy"
    assert getRefFluxField(schema, filt) == f"{filt}_flux"


def test_schema_covers_the_filters_the_task_asks_for():
    from lsst.ts.wep.blitz.donutBlitzCorner import DonutBlitzCornerConfig

    config = DonutBlitzCornerConfig()
    # photoRefFilter, when set, replaces the prefix path entirely -- so the prefix
    # below is only the filter the task will actually ask for while it stays None.
    assert config.photoRefFilter is None
    wanted = {config.astromRefFilter}
    wanted.update(f"{config.photoRefFilterPrefix}_{band}" for band in "ugrizy")
    assert wanted <= set(refcat_store.FLUX_FILTERS)


# One level-5 file, split into its level-7 children, for the reshard tests.
RESHARD_SHARD_ID = 12345


@pytest.fixture(scope="module")
def shards():
    if not SHARD_PATHS:
        pytest.skip(NO_SHARDS)
    return refcat_store.load_and_reshard([RESHARD_SHARD_ID])


def test_children_are_exactly_the_16_of_the_parent(shards):
    first = RESHARD_SHARD_ID * 4 ** (refcat_store.LOAD_LEVEL - refcat_store.SHARD_LEVEL)
    assert sorted(shards) == list(range(first, first + 16))


def test_no_rows_are_lost(shards):
    from astropy.io import fits

    source = fits.getdata(os.path.join(SHARD_DIR, f"{RESHARD_SHARD_ID}.fits"))
    assert sum(len(catalog) for catalog in shards.values()) == len(source)


def test_every_row_lands_in_its_own_pixel(shards):
    for index, catalog in shards.items():
        pixel = HtmPixelization(refcat_store.LOAD_LEVEL).pixel(index)
        for record in catalog:
            assert pixel.contains(UnitVector3d(record.getCoord().getVector()))


def test_fluxes_survive_the_astropy_round_trip(shards):
    # SimpleCatalog.readFits on these files silently drops the nine flux/error
    # columns; this is the guard that we are not doing that. Pick the biggest
    # child so the assertion can't fail merely because a shard came out empty.
    catalog = max(shards.values(), key=len)
    for filt in refcat_store.FLUX_FILTERS:
        assert np.isfinite(catalog[f"{filt}_flux"]).any()


def test_catalogs_carry_a_refcat_format_version(shards):
    from lsst.meas.algorithms.loadReferenceObjects import getFormatVersionFromRefCat

    assert getFormatVersionFromRefCat(max(shards.values(), key=len)) == 2


@pytest.fixture(autouse=True)
def clean_shard_cache():
    """_SHARD_CACHE is module-level and shared across stores, so a test that
    asserts on what a fresh store had to load is otherwise order-dependent."""
    refcat_store._SHARD_CACHE.clear()
    yield
    refcat_store._SHARD_CACHE.clear()


@pytest.mark.skipif(not SHARD_PATHS, reason=NO_SHARDS)
def test_store_reuses_on_a_repeat_pointing():
    store = refcat_store.RefCatStore()
    first = store.ensure(*BORESIGHT)
    assert not first["reused"]

    shards = store.shards
    # Keyed on the shard set, not the boresight, so a small dither still hits.
    second = store.ensure(BORESIGHT[0] + 0.001, BORESIGHT[1])
    assert second["reused"]
    assert store.shards is shards


@pytest.mark.skipif(not SHARD_PATHS, reason=NO_SHARDS)
def test_a_slew_loads_only_the_shards_it_does_not_already_have():
    """The point of the module-level shard cache: a new shard set is a cold path
    for the store but should still only read the parents nobody has read yet."""
    slewed = (BORESIGHT[0] + 2.0, BORESIGHT[1])
    overlap = set(refcat_store.shard_ids_for_pointing(*BORESIGHT)) & set(
        refcat_store.shard_ids_for_pointing(*slewed)
    )
    assert overlap, "pick a slew that actually shares some shards"

    store = refcat_store.RefCatStore()
    first = store.ensure(*BORESIGHT)
    assert first["n_shards_loaded"] == first["n_level5_files"]

    second = store.ensure(*slewed)
    assert not second["reused"]  # the set really did change
    assert second["n_shards_reused"] == len(overlap)
    assert second["n_shards_loaded"] == second["n_level5_files"] - len(overlap)


@pytest.mark.skipif(not SHARD_PATHS, reason=NO_SHARDS)
def test_a_second_store_at_the_same_pointing_reads_nothing():
    """Each PreparedEntry builds its own RefCatStore, so the cache is only worth
    having if it is shared between instances."""
    refcat_store.RefCatStore().ensure(*BORESIGHT)

    fresh = refcat_store.RefCatStore().ensure(*BORESIGHT)

    assert not fresh["reused"]  # a fresh store's own shard set was empty
    assert fresh["n_shards_loaded"] == 0
    assert fresh["n_shards_reused"] == fresh["n_level5_files"]


@pytest.mark.skipif(not SHARD_PATHS, reason=NO_SHARDS)
def test_shard_cache_cap_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(refcat_store, "SHARD_CACHE_CAP", 3)

    refcat_store._ensure_shards(frozenset({12345, 12346}))
    refcat_store._ensure_shards(frozenset({12345, 12347}))  # touches 12345
    refcat_store._ensure_shards(frozenset({12348}))  # over cap: drops 12346

    assert set(refcat_store._SHARD_CACHE) == {12345, 12347, 12348}


@pytest.mark.skipif(not SHARD_PATHS, reason=NO_SHARDS)
def test_a_wanted_set_larger_than_the_cap_is_still_returned_whole(monkeypatch):
    """Eviction runs after composition, so asking for more parents than the cap
    holds must not lose children on the way out."""
    monkeypatch.setattr(refcat_store, "SHARD_CACHE_CAP", 1)
    wanted = frozenset({12345, 12346})

    shards, loaded = refcat_store._ensure_shards(wanted)

    assert loaded == 2
    assert len(shards) == 32  # 16 level-7 children per level-5 parent
    assert len(refcat_store._SHARD_CACHE) == 1


@pytest.mark.skipif(not SHARD_PATHS, reason=NO_SHARDS)
def test_store_reports_uncovered_shards():
    store = refcat_store.RefCatStore()
    store.ensure(*BORESIGHT)
    assert store.uncovered(set(EXPECTED_SHARDS)) == set()
    assert store.uncovered({8192}) == {8192}
