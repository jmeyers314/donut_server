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

SHARD_PATHS = sorted(glob.glob(os.path.join(refcat_store.REFCAT_DIR, "*.fits")))
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
RAW_PATHS = sorted(glob.glob(os.path.join(RAW_DIR, "raw_*_r.fits")))

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
        pytest.skip(f"no raw_*.fits in {RAW_DIR}")

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
        pytest.skip(f"no shards in {refcat_store.REFCAT_DIR}")
    return refcat_store.load_and_reshard([RESHARD_SHARD_ID])


def test_children_are_exactly_the_16_of_the_parent(shards):
    first = RESHARD_SHARD_ID * 4 ** (refcat_store.LOAD_LEVEL - refcat_store.SHARD_LEVEL)
    assert sorted(shards) == list(range(first, first + 16))


def test_no_rows_are_lost(shards):
    from astropy.io import fits

    source = fits.getdata(os.path.join(refcat_store.REFCAT_DIR, f"{RESHARD_SHARD_ID}.fits"))
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


@pytest.mark.skipif(not SHARD_PATHS, reason=f"no shards in {refcat_store.REFCAT_DIR}")
def test_store_reuses_on_a_repeat_pointing():
    store = refcat_store.RefCatStore()
    first = store.ensure(*BORESIGHT)
    assert not first["reused"]

    shards = store.shards
    # Keyed on the shard set, not the boresight, so a small dither still hits.
    second = store.ensure(BORESIGHT[0] + 0.001, BORESIGHT[1])
    assert second["reused"]
    assert store.shards is shards


@pytest.mark.skipif(not SHARD_PATHS, reason=f"no shards in {refcat_store.REFCAT_DIR}")
def test_store_reports_uncovered_shards():
    store = refcat_store.RefCatStore()
    store.ensure(*BORESIGHT)
    assert store.uncovered(set(EXPECTED_SHARDS)) == set()
    assert store.uncovered({8192}) == {8192}
