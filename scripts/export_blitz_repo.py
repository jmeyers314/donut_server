#!/usr/bin/env python
"""Export everything DonutBlitzCornerTask needs, at USDF, into one directory.

Run this at USDF (`ssh s3df`), then rsync the output directory to a laptop and
feed it to `build_blitz_repo.py`. Nothing here writes to the source repo.

Why a script instead of the stock CLI: `butler export-calibs` walks *every*
CALIBRATION collection and takes every dataset of every calibration type, with
no detector or filter restriction. For LSSTCam that is the whole 189-detector
focal plane across all physical filters -- orders of magnitude more than the
eight corner wavefront sensors this task reads. So the dataset queries are
spelled out here and restricted to the corners and to the bands the requested
visits actually use.

Certification rides along for free: `saveCollection()` on a CALIBRATION
collection exports its dataset associations *with their validity timespans*,
and `butler import` replays them through `registry.certify()`. There is no
separate `certify-calibrations` step on the destination side.

The refcat is exported as the real `the_monster_20250219` htm7 shards, but only
those overlapping the visits -- a 1.8 deg field touches ~55 of the 131072
shards, so this is a few hundred MB rather than the ~880 GB full catalog.

Raws are handled outside the YAML export, in a `raw_zips/` subdirectory. Each
LSSTCam raw dataset in `/repo/main` is stored as one artifact per exposure
holding every detector's FITS file, addressed internally by a `zip-path=` URI
fragment. `butler.export()`/`butler import` copy that artifact as if it were
a single-file dataset and lose the fragment, so the destination formatter sees
a bare `.zip` and rejects it. `retrieve_artifacts_zip()` is the API that
understands this representation: it writes a self-indexed Zip per exposure
that `butler ingest-zip` (run by `build_blitz_repo.py`) can register directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from lsst.daf.butler import Butler, CollectionType, DatasetRef

# The eight corner wavefront sensors. SW0 is extra-focal, SW1 intra-focal.
CORNER_DETECTORS = (191, 192, 195, 196, 199, 200, 203, 204)

# Per-detector calibs: no filter dependence, so one query each.
DETECTOR_CALIBS = ("ptc", "linearizer", "crosstalk")

# Calibs that also depend on physical_filter.
FILTER_CALIBS = ("flat", "intrinsicZernikes")

REFCAT = "the_monster_20250219"
REFCAT_DIMENSION = "htm7"

# Padding on the focal-plane radius when choosing refcat shards. The task's own
# loader pads the detector bounding boxes by a margin before taking their HTM
# envelope; a whole-field radius with margin is a strict superset of that, and
# over-exporting a shard is cheap while missing one is a task failure.
FIELD_RADIUS_DEG = 1.85

log = logging.getLogger("export_blitz_repo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", default="/repo/main", help="Source butler repo or alias (default: /repo/main)."
    )
    parser.add_argument(
        "--collections",
        default="LSSTCam/defaults",
        help="Input collection to resolve raws and calibs from (default: LSSTCam/defaults).",
    )
    parser.add_argument(
        "--instrument", default="LSSTCam", help="Instrument name (default: LSSTCam)."
    )
    parser.add_argument(
        "--visit",
        type=int,
        nargs="+",
        action="extend",
        required=True,
        dest="visits",
        help="Visit(s) to export. Repeat and/or pass several values to one flag.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory. Must not already exist -- butler.export() refuses to "
        "merge into a populated tree, and a half-overwritten export is worse than none.",
    )
    parser.add_argument(
        "--transfer",
        default="copy",
        choices=("copy", "auto", "link", "symlink", "hardlink", "relsymlink", "direct"),
        help="How to move file artifacts into --output. Default 'copy', which is the only "
        "mode that survives being rsynced to another machine.",
    )
    parser.add_argument(
        "--skip-refcat",
        action="store_true",
        help="Skip the reference catalog. Only useful if you already have one locally.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts and exit.")
    return parser.parse_args(argv)


def find_refcat_shards(butler: Butler, visits: list[int], instrument: str) -> set[int]:
    """htm7 shard indices overlapping the given visits.

    Uses each visit's own boresight from its exposure records rather than its
    `visit.region`: region is populated by `define-visits` and may be absent,
    while tracking_ra/tracking_dec are written at ingest and always present.
    """
    from lsst.sphgeom import Angle, Circle, HtmPixelization, LonLat, UnitVector3d

    pixelization = HtmPixelization(int(REFCAT_DIMENSION.removeprefix("htm")))

    shards: set[int] = set()
    for visit in visits:
        records = butler.query_dimension_records(
            "exposure",
            where="instrument = :inst AND visit = :v",
            bind={"inst": instrument, "v": visit},
            explain=False,
        )
        if not records:
            raise RuntimeError(
                f"No exposure records for visit {visit}. Either the visit is not in "
                f"{butler.collections.defaults!r} or visits are not defined for it."
            )
        record = records[0]
        if record.tracking_ra is None or record.tracking_dec is None:
            raise RuntimeError(f"Visit {visit} exposure record has no boresight.")
        center = UnitVector3d(LonLat.fromDegrees(record.tracking_ra, record.tracking_dec))
        circle = Circle(center, Angle.fromDegrees(FIELD_RADIUS_DEG))
        for begin, end in pixelization.envelope(circle):
            shards.update(range(begin, end))
        log.info(
            "Visit %d boresight (%.4f, %.4f)", visit, record.tracking_ra, record.tracking_dec
        )
    return shards


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if os.path.exists(args.output):
        log.error("Output directory already exists: %s", args.output)
        return 1

    butler = Butler.from_config(args.repo, collections=args.collections)
    visits = sorted(set(args.visits))
    log.info("Source %s, collection %s, visits %s", args.repo, args.collections, visits)

    # --- raws -------------------------------------------------------------
    # Exported with dimension records so the destination gets the exposure,
    # group, day_obs and physical_filter rows that `define-visits` will need.
    raw_refs: list[DatasetRef] = []
    for visit in visits:
        refs = butler.query_datasets(
            "raw",
            collections=args.collections,
            where="instrument = :inst AND visit = :v",
            bind={"inst": args.instrument, "v": visit},
            with_dimension_records=True,
            find_first=True,
            explain=False,
            limit=None,
        )
        # Filter detectors in Python: `detector IN (...)` with a bound sequence is
        # not portable across butler versions, and the per-visit row count is tiny.
        kept = [r for r in refs if r.dataId["detector"] in CORNER_DETECTORS]
        if not kept:
            raise RuntimeError(
                f"Visit {visit} yielded no corner-sensor raws "
                f"(saw {len(refs)} raws across all detectors)."
            )
        log.info("Visit %d: %d corner raws", visit, len(kept))
        raw_refs.extend(kept)

    # Bands present in the exported raws drive which flats are needed. Querying
    # all six would export five times the flats for no benefit.
    physical_filters = sorted({str(r.dataId["physical_filter"]) for r in raw_refs})
    log.info("Physical filters in play: %s", physical_filters)

    # --- calibs -----------------------------------------------------------
    calib_refs: list[DatasetRef] = []
    calib_collections: set[str] = set()

    # Only CALIBRATION collections carry validity ranges worth re-certifying.
    for info in butler.collections.query_info(
        args.collections,
        flatten_chains=True,
        include_chains=True,
        collection_types={CollectionType.CALIBRATION},
    ):
        calib_collections.add(info.name)
    log.info("Calibration collections under %s: %s", args.collections, sorted(calib_collections))

    # One registry hit per visit rather than one per (type, detector, filter, visit).
    visit_timespans = {v: _visit_timespan(butler, args.instrument, v) for v in visits}

    for dataset_type in DETECTOR_CALIBS + FILTER_CALIBS:
        by_filter = dataset_type in FILTER_CALIBS
        found = 0
        for detector in CORNER_DETECTORS:
            targets = physical_filters if by_filter else [None]
            for physical_filter in targets:
                data_id = {"instrument": args.instrument, "detector": detector}
                if physical_filter is not None:
                    data_id["physical_filter"] = physical_filter
                # find_dataset resolves a calib through the validity ranges for a
                # specific time. Each visit can in principle land in a different
                # validity window, so ask per visit and de-duplicate.
                for visit, timespan in visit_timespans.items():
                    try:
                        ref = butler.find_dataset(
                            dataset_type,
                            data_id,
                            collections=args.collections,
                            timespan=timespan,
                            dimension_records=True,
                        )
                    except Exception as exc:  # unregistered type, etc.
                        log.warning("%s: %s", dataset_type, exc)
                        ref = None
                    if ref is not None:
                        calib_refs.append(ref)
                        found += 1
        if found:
            log.info("%s: %d refs (pre-dedup)", dataset_type, found)
        else:
            # intrinsicZernikes has minimum=0 on the connection, so it is not
            # fatal; the others are.
            level = log.warning if dataset_type == "intrinsicZernikes" else log.error
            level("%s: nothing found for the corner sensors", dataset_type)

    calib_refs = list(set(calib_refs))
    log.info("Unique calibration datasets: %d", len(calib_refs))

    # --- refcat -----------------------------------------------------------
    refcat_refs: list[DatasetRef] = []
    if not args.skip_refcat:
        shards = find_refcat_shards(butler, visits, args.instrument)
        log.info("Refcat shards overlapping the visits: %d", len(shards))
        for shard in sorted(shards):
            ref = butler.find_dataset(
                REFCAT,
                {REFCAT_DIMENSION: shard},
                collections=args.collections,
                dimension_records=True,
            )
            # A shard with no sources simply has no dataset; that is normal near
            # the poles and in masked regions, so a miss is not an error.
            if ref is not None:
                refcat_refs.append(ref)
        log.info("Refcat datasets found: %d of %d shards", len(refcat_refs), len(shards))
        if not refcat_refs:
            log.error("No %s datasets found -- is it in %s?", REFCAT, args.collections)

    total = len(raw_refs) + len(calib_refs) + len(refcat_refs)
    log.info(
        "Totals: %d raw, %d calib, %d refcat = %d datasets",
        len(raw_refs),
        len(calib_refs),
        len(refcat_refs),
        total,
    )
    if args.dry_run:
        log.info("Dry run, nothing written.")
        return 0

    # --- export -----------------------------------------------------------
    # Raws are excluded from the YAML export below. `/repo/main` stores each
    # LSSTCam raw as one artifact per exposure holding every detector's FITS
    # file, addressed internally by a `zip-path=` URI fragment.
    # `butler.export()`/`butler import` don't preserve that fragment -- they
    # copy the artifact as if it were a single-file dataset, so the
    # destination formatter sees a bare `.zip` and rejects it. `retrieve_artifacts_zip`
    # is the API that understands this: it writes a self-contained Zip with
    # its own embedded index, which `butler ingest-zip` (used by
    # build_blitz_repo.py) can register directly. One call per exposure,
    # since a Zip's contents must share a dataset type but can span detectors.
    os.makedirs(args.output)
    raw_zip_dir = os.path.join(args.output, "raw_zips")
    os.makedirs(raw_zip_dir)
    raw_refs_by_exposure: dict[int, list[DatasetRef]] = {}
    for ref in raw_refs:
        raw_refs_by_exposure.setdefault(ref.dataId["exposure"], []).append(ref)
    log.info(
        "Retrieving %d raw datasets as %d exposure Zips", len(raw_refs), len(raw_refs_by_exposure)
    )
    for exposure, refs in sorted(raw_refs_by_exposure.items()):
        zip_path = butler.retrieve_artifacts_zip(refs, destination=raw_zip_dir)
        log.info("Exposure %d: %d raws -> %s", exposure, len(refs), zip_path)

    log.info("Starting export to %s", args.output)
    with butler.export(directory=args.output, format="yaml", transfer=args.transfer) as export:
        # Collections first. Exporting a CALIBRATION collection carries the
        # dataset<->timespan associations, which import replays via certify().
        log.info("Saving %d calibration collections", len(calib_collections))
        for name in sorted(calib_collections):
            try:
                export.saveCollection(name)
            except Exception as exc:
                log.warning("Could not save collection %s: %s", name, exc)

        # Raw datasets themselves are not saved here (see raw_zips above),
        # but their dimension records are, since `define-visits` on the
        # destination needs the exposure/group/day_obs/physical_filter rows.
        log.info("Saving dimension records for %d raw data IDs", len(raw_refs))
        export.saveDataIds([r.dataId for r in raw_refs])
        log.info("Saving %d calib datasets", len(calib_refs))
        export.saveDatasets(calib_refs)
        if refcat_refs:
            log.info("Saving %d refcat datasets", len(refcat_refs))
            export.saveDatasets(refcat_refs)

    log.info("Wrote export to %s", args.output)
    log.info("Now: rsync -a --info=progress2 %s <laptop>:<dest>/", args.output.rstrip("/"))
    return 0


def _visit_timespan(butler: Butler, instrument: str, visit: int):
    """Timespan of a visit, for resolving calibration validity ranges."""
    records = butler.query_dimension_records(
        "visit",
        where="instrument = :inst AND visit = :v",
        bind={"inst": instrument, "v": visit},
        explain=False,
    )
    if not records:
        raise RuntimeError(f"No visit record for {visit}")
    return records[0].timespan


if __name__ == "__main__":
    sys.exit(main())
