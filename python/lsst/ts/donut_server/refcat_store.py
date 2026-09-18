"""Reference catalog shards for the wavefront pipeline.

The task's `refCat` connection declares `dimensions=("htm7",)`, but the local
extraction in `ref_cat/` is sharded at HTM **level 5**: file `N.fits` holds
exactly level-5 pixel `N`, i.e. exactly the 16 level-7 children
`N*16 .. N*16+15`. Ids run 8192..16383, which is the complete level-5 range
(`8*4^5 .. 16*4^5-1`), so the local set covers the whole sky and a shard can
never be missing.

This module bridges that gap by resharding to level 7 **in memory**, at prepare
time, from the pointing alone. Two facts about the local files force the shape
of the code:

1. **They are not afw-format refcats.** They are plain BinTables with no afw
   schema headers, no TUNIT and no REFCAT_FORMAT_VERSION, so
   `SimpleCatalog.readFits` silently returns only `id, coord_ra, coord_dec` and
   drops the nine flux/error columns. They have to be read with astropy and
   assembled into `SimpleCatalog`s against a schema built here.
2. **`InMemoryLimitedButler` deep-copies on every `get()`**, and
   `ReferenceObjectLoader` calls `.get()` once per overlapping shard *per
   detector*. Level-5 shards are large (up to 1.19M rows, 629 ms to deep-copy),
   so handing them over directly costs 1268 ms of the push critical path.
   Resharding to level 7 gives bit-identical output for 118 ms, and moves the
   read and rebuild into prepare, where there is slack.

Imported only by `coordinator.py` -- the FastAPI process must import no LSST
code.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
from astropy.io import fits

import esutil.htm
import lsst.afw.table as afwTable
import lsst.geom as geom
from lsst.daf.butler import DataCoordinate
from lsst.meas.algorithms.convertReferenceCatalog import addRefCatMetadata
from lsst.sphgeom import Angle, Circle, ConvexPolygon, HtmPixelization, LonLat, UnitVector3d

REFCAT_DIR = os.path.join(os.path.dirname(__file__), "ref_cat")

SHARD_LEVEL = 5

# What we reshard to before handing shards to the loader. Must equal the level in
# the refCat connection's declared dimensions -- guarded by a test, since a
# silent mismatch would make every DatasetRef unbuildable (or, worse, buildable
# with the wrong region).
LOAD_LEVEL = 7
LOAD_DIMENSION = f"htm{LOAD_LEVEL}"

# What `loadPixelBox` really queries, and therefore what the push-time coverage
# check has to match. Both terms are needed: the task sets `pixelMargin = 300` as
# a bare literal in its run() (donutBlitzCorner.py), `loadPixelBox` grows
# the bbox by that, and then `_makeBoxRegion` grows the *outer* region -- the one
# shard selection uses -- by another `bboxToSpherePadding=100`. Checking only the
# 300 lets the guard pass while the loader reaches into an unloaded shard.
TASK_PIXEL_MARGIN_PX = 300
LOADER_BBOX_PADDING_PX = 100
COVERAGE_MARGIN_PX = TASK_PIXEL_MARGIN_PX + LOADER_BBOX_PADDING_PX

# Measured max angular distance from boresight to any corner-sensor pixel, with
# COVERAGE_MARGIN_PX included: 1.878 deg. The slack over that costs 2 extra
# level-5 shards (10 rather than the 8 strictly needed) and buys tolerance for
# dithers between prepare and push. Guarded by a test against the real raws.
FIELD_RADIUS_DEG = 2.0

# The task reads astromRefFilter ("phot_g_mean") for the WCS fit and
# "{photoRefFilterPrefix}_{band}" ("monster_ComCam_r" etc.) for donut selection.
# These are the flux columns present in the local extraction; upstream notes
# there are no monster_LSSTCam_* columns in any released refcat version. Not
# derivable from the task config: this schema is built once for all bands.
FLUX_FILTERS = ("phot_g_mean",) + tuple(f"monster_ComCam_{b}" for b in "ugrizy")

_SCHEMA: Any = None


def schema() -> Any:
    """The one schema every shard catalog is built against.

    A single shared object, not an equal-valued copy per shard: `loadRegion`
    raises `TypeError("Reference catalogs have mismatching schemas")` on any
    mismatch between the first shard and the rest.
    """
    global _SCHEMA
    if _SCHEMA is None:
        sch = afwTable.SimpleTable.makeMinimalSchema()
        # Yields coord_raErr/coord_decErr as float32 in rad, matching what the
        # real refcat writer produces.
        afwTable.CovarianceMatrix2fKey.addFields(
            schema=sch, prefix="coord", names=["ra", "dec"], units=["rad", "rad"],
            diagonalOnly=True,
        )
        for filt in FLUX_FILTERS:
            # units="nJy" is load-bearing: the loader only consults
            # getFormatVersionFromRefCat() when the flux units are *not* nJy, and
            # that raises on these files. addRefCatMetadata below covers the same
            # ground, so neither is load-bearing alone.
            sch.addField(f"{filt}_flux", type=np.float64, units="nJy", doc=f"{filt} flux")
        _SCHEMA = sch
    return _SCHEMA


def _range_set_ids(range_set) -> list[int]:
    """Flatten a sphgeom RangeSet of half-open (begin, end) pairs."""
    ids: list[int] = []
    for begin, end in range_set:
        ids.extend(range(begin, end))
    return ids


def shard_ids_for_pointing(
    ra_deg: float, dec_deg: float, radius_deg: float = FIELD_RADIUS_DEG
) -> list[int]:
    """Level-5 shard ids covering a circle of `radius_deg` about a boresight.

    Rotator angle is deliberately not a parameter: the region is a circle
    centred on the boresight, so rotation cannot change which shards it touches.
    """
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        raise ValueError(f"non-finite boresight: ra={ra_deg!r} dec={dec_deg!r}")
    if not -90.0 <= dec_deg <= 90.0:
        raise ValueError(f"boresight dec out of range: {dec_deg!r}")
    if not radius_deg > 0.0:
        raise ValueError(f"radius must be positive: {radius_deg!r}")

    circle = Circle(
        UnitVector3d(LonLat.fromDegrees(ra_deg, dec_deg)), Angle.fromDegrees(radius_deg)
    )
    return sorted(_range_set_ids(HtmPixelization(SHARD_LEVEL).envelope(circle)))


def shard_ids_for_exposures(exposures: dict[str, Any]) -> set[int]:
    """Level-5 shard ids the raws actually need, from their own WCSs.

    Used at push time to check that prepare loaded a covering set. Exact where an
    angular tolerance on the boresight would be a guess, and it costs ~0.3 ms.
    The `ConvexPolygon` of the padded box corners is the same construction
    `ReferenceObjectLoader._makeBoxRegion` uses, so the two regions agree.
    """
    pixelization = HtmPixelization(SHARD_LEVEL)
    ids: set[int] = set()
    for exp in exposures.values():
        box = geom.Box2D(exp.getBBox())
        box.grow(COVERAGE_MARGIN_PX)
        wcs = exp.getWcs()
        vertices = [
            UnitVector3d(wcs.pixelToSky(corner).getVector()) for corner in box.getCorners()
        ]
        ids.update(_range_set_ids(pixelization.envelope(ConvexPolygon.convexHull(vertices))))
    return ids


def htm_data_id(universe: Any, index: int) -> Any:
    """A skypix DataCoordinate that actually carries its region.

    Unlike every other dataId the coordinator builds, refCat's has to carry a
    dimension *record*: `ReferenceObjectLoader` filters shards on
    `dataId.region`, and `DataCoordinate.standardize` alone leaves
    `hasRecords()` False and `region` None, which surfaces as
    `AttributeError: 'NoneType' object has no attribute 'intersects'`. skypix
    regions are pure geometry, so the record needs no Registry to build.

    The region comes off the universe's own pixelization rather than a locally
    constructed one, so it cannot disagree with the dimension it is keyed by.
    """
    dimension = universe[LOAD_DIMENSION]
    record = dimension.RecordClass(id=index, region=dimension.pixelization.pixel(index))
    return DataCoordinate.standardize(
        {LOAD_DIMENSION: index}, universe=universe
    ).expanded({LOAD_DIMENSION: record})


def _build_catalog(columns: dict[str, np.ndarray], mask: np.ndarray) -> Any:
    """One level-7 shard catalog, from the rows of its level-5 parent."""
    catalog = afwTable.SimpleCatalog(schema())
    catalog.resize(int(mask.sum()))
    for name, values in columns.items():
        catalog[name] = values[mask]
    addRefCatMetadata(catalog)
    return catalog


def load_and_reshard(shard_ids: list[int]) -> dict[int, Any]:
    """Read level-5 files and split them into one catalog per level-7 child.

    Returns `{level-7 index: SimpleCatalog}`. Level-7 indices come from
    `esutil.htm`, which is vectorised over whole columns; `lsst.sphgeom`'s
    `HtmPixelization.index` is per-point and would be far too slow at ~1M rows.
    It takes degrees, hence the rad2deg -- the stored values are radians.
    """
    indexer = esutil.htm.HTM(LOAD_LEVEL)
    shards: dict[int, Any] = {}
    for shard_id in shard_ids:
        path = os.path.join(REFCAT_DIR, f"{shard_id}.fits")
        if not os.path.exists(path):
            # The whole sky is covered when the directory is complete, so a miss
            # means the local copy is partial -- say that rather than letting a
            # bare FileNotFoundError imply the pointing was wrong.
            raise RuntimeError(
                f"{REFCAT_DIR} is incomplete: no shard {shard_id} at {path}. "
                f"Expected all 8192 level-{SHARD_LEVEL} files, ids 8192..16383."
            )
        record = fits.getdata(path)
        columns = {name: np.asarray(record[name]) for name in record.dtype.names}
        child_ids = indexer.lookup_id(
            np.rad2deg(columns["coord_ra"]), np.rad2deg(columns["coord_dec"])
        )
        for child_id in np.unique(child_ids):
            shards[int(child_id)] = _build_catalog(columns, child_ids == child_id)
    return shards


class RefCatStore:
    """Cache of level-7 shard catalogs for the current pointing.

    Keyed on the *set of level-5 shard ids*, not on the boresight itself, so an
    AOS sequence dithering within one pointing keeps hitting the cache.
    """

    def __init__(self) -> None:
        # shard_ids records what was *requested*, which is what `uncovered` has to
        # compare against -- deriving it from `shards` (via `index >> 4`) would
        # report a requested-but-empty shard as uncovered.
        self.shard_ids: frozenset[int] = frozenset()
        self.shards: dict[int, Any] = {}
        self.n_rows = 0

    def ensure(self, ra_deg: float, dec_deg: float) -> dict:
        t0 = time.monotonic()
        wanted = frozenset(shard_ids_for_pointing(ra_deg, dec_deg))
        reused = wanted == self.shard_ids

        if not reused:
            shards = load_and_reshard(sorted(wanted))
            # Rebind rather than mutate, so a failure above leaves the previous
            # pointing's cache intact instead of half-replaced.
            self.shards = shards
            self.shard_ids = wanted
            self.n_rows = sum(len(catalog) for catalog in shards.values())

        # Same keys either way, so /prepare's response shape does not change
        # between a cold call and a reused one.
        return {
            "reused": reused,
            "elapsed_s": time.monotonic() - t0,
            "n_level5_files": len(self.shard_ids),
            "n_level7_shards": len(self.shards),
            "n_rows": self.n_rows,
            "boresight": [ra_deg, dec_deg],
        }

    def uncovered(self, needed: set[int]) -> set[int]:
        """Level-5 ids in `needed` that this store was not built for."""
        return needed - self.shard_ids
