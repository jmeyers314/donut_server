#!/usr/bin/env python
"""Build a local butler repo from an export made by export_blitz_repo.py.

Run this on the laptop, on a directory rsynced down from USDF. The result is a
repo that `pipetask run -t ...DonutBlitzCornerTask` can read.

Steps, in order, and why each is needed:

1. `butler create` -- empty repo, SQLite registry.
2. `register-instrument LSSTCam` -- populates the detector, physical_filter and
   band dimension tables. Must precede import, since the imported datasets
   reference those rows.
3. `write-curated-calibrations` -- camera geometry and defects. `define-visits`
   needs camera geometry to compute visit_detector_region.
4. `butler import` -- calib and refcat datasets, plus dimension records for
   raws and everything else. Calibration collections come back certified:
   import replays the exported validity timespans through
   `registry.certify()`, so no certify step is needed here.
5. `butler ingest-zip` -- the raw datasets themselves, one Zip per exposure.
   Raws are not in export.yaml: `/repo/main` stores each LSSTCam raw as one
   artifact per exposure holding every detector's FITS file, addressed
   internally by a `zip-path=` URI fragment, and `butler import` cannot
   preserve that fragment. `export_blitz_repo.py` instead wrote each
   exposure's raws as a self-indexed Zip under `raw_zips/`, which
   `ingest-zip` reads directly.
6. `define-visits` -- builds the visit, visit_definition and
   visit_detector_region rows from the imported exposures. The task's quantum
   dimensions are (instrument, visit), so without this there is nothing to run
   against.
7. `collection-chain LSSTCam/defaults` -- one input collection covering raws,
   calibs and refcats, matching how /repo/main is arranged.

The skymap steps that appear in older repo-setup notes are deliberately absent:
DonutBlitzCornerTask's dimensions are (instrument, visit) plus htm7, so it never
touches a skymap.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys

INSTRUMENT_CLASS = "lsst.obs.lsst.LsstCam"
INSTRUMENT = "LSSTCam"
DEFAULTS_CHAIN = "LSSTCam/defaults"

# Curated calibrations go in their own collection. This stack's
# `write-curated-calibrations` requires an explicit --collection (or a label),
# and naming it here keeps it distinct from the imported calibs: curated
# includes its own `crosstalk`, and the chain order below has to put the
# imported one first so it is the one the task resolves.
CURATED_COLLECTION = "LSSTCam/calib/curated"

# Where export_blitz_repo.py writes the per-exposure raw Zips, relative to
# the export directory. Raws never appear in export.yaml; each Zip carries
# its own index, so `butler ingest-zip` registers it without one.
RAW_ZIP_GLOB = "raw_zips/*.zip"

log = logging.getLogger("build_blitz_repo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export_dir", help="Directory rsynced from USDF (holds export.yaml).")
    parser.add_argument("repo", help="Path of the repo to create.")
    parser.add_argument(
        "--transfer",
        default="copy",
        choices=("copy", "auto", "link", "symlink", "hardlink", "relsymlink", "direct"),
        help="How to bring artifacts into the repo. 'copy' makes the repo standalone so the "
        "export directory can be deleted afterwards; 'direct' leaves it depending on it.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue past a failing step instead of stopping. For debugging only.",
    )
    return parser.parse_args(argv)


def run(cmd: list[str], *, keep_going: bool) -> int:
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Command failed with exit code %d", result.returncode)
        if not keep_going:
            raise SystemExit(result.returncode)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    export_yaml = os.path.join(args.export_dir, "export.yaml")
    if not os.path.exists(export_yaml):
        log.error("No export.yaml in %s -- is that the rsynced export directory?", args.export_dir)
        return 1
    raw_zips = sorted(glob.glob(os.path.join(args.export_dir, RAW_ZIP_GLOB)))
    if not raw_zips:
        log.error("No raw Zips found under %s -- is that the rsynced export directory?", args.export_dir)
        return 1
    if os.path.exists(args.repo):
        log.error("Repo path already exists: %s", args.repo)
        return 1

    keep = args.keep_going

    run(["butler", "create", args.repo], keep_going=keep)
    run(["butler", "register-instrument", args.repo, INSTRUMENT_CLASS], keep_going=keep)
    run(
        [
            "butler",
            "write-curated-calibrations",
            args.repo,
            INSTRUMENT_CLASS,
            "--collection",
            CURATED_COLLECTION,
        ],
        keep_going=keep,
    )
    run(
        [
            "butler",
            "import",
            args.repo,
            args.export_dir,
            "--export-file",
            export_yaml,
            "--transfer",
            args.transfer,
        ],
        keep_going=keep,
    )

    # Raws are not in export.yaml (see module docstring); each Zip is
    # self-indexed, so no --export-file is needed here.
    log.info("Ingesting %d raw Zips", len(raw_zips))
    for zip_path in raw_zips:
        run(
            ["butler", "ingest-zip", args.repo, zip_path, "--transfer", args.transfer],
            keep_going=keep,
        )

    run(["butler", "define-visits", args.repo, INSTRUMENT_CLASS], keep_going=keep)

    # Chain the imported collections into one input. Discovered rather than
    # hardcoded: the calib collection name on the source repo is whatever the
    # export found, and RUN names are generated at ingest time.
    children = discover_children(args.repo)
    log.info("Chaining into %s: %s", DEFAULTS_CHAIN, children)
    if children:
        run(
            ["butler", "collection-chain", args.repo, DEFAULTS_CHAIN, "--mode", "redefine", *children],
            keep_going=keep,
        )

    log.info("Repo built at %s", args.repo)
    report(args.repo)
    return 0


def discover_children(repo: str) -> list[str]:
    """Collections to put in the defaults chain, highest priority first.

    Ordering matters in two ways:

    * CALIBRATION collections precede RUNs, so a validity-range lookup wins over
      a bare RUN hit for the same dataset.
    * The *imported* calibration collection precedes the curated one. Both define
      `crosstalk`, and the one exported from the source repo is the one that was
      actually used there, so it must shadow the obs_lsst-curated default.
    """
    from lsst.daf.butler import Butler, CollectionType

    butler = Butler.from_config(repo)
    imported_calib, curated_calib, run_collections = [], [], []
    for info in butler.collections.query_info("*", include_chains=False):
        if info.name == DEFAULTS_CHAIN:
            continue
        if info.type == CollectionType.CALIBRATION:
            if info.name == CURATED_COLLECTION:
                curated_calib.append(info.name)
            else:
                imported_calib.append(info.name)
        elif info.type == CollectionType.RUN:
            run_collections.append(info.name)
    return sorted(imported_calib) + sorted(curated_calib) + sorted(run_collections)


def report(repo: str) -> None:
    """Print what landed, so a missing input is obvious before pipetask runs."""
    from lsst.daf.butler import Butler

    butler = Butler.from_config(repo, collections=DEFAULTS_CHAIN)

    visits = butler.query_dimension_records("visit", explain=False)
    log.info("Visits defined: %s", sorted(v.id for v in visits))

    expected = [
        "raw",
        "ptc",
        "linearizer",
        "crosstalk",
        "flat",
        "intrinsicZernikes",
        "the_monster_20250219",
    ]
    registered = {dt.name for dt in butler.registry.queryDatasetTypes(...)}
    for name in expected:
        if name not in registered:
            log.warning("%-22s NOT REGISTERED", name)
            continue
        refs = butler.query_datasets(
            name, collections=DEFAULTS_CHAIN, find_first=False, explain=False, limit=None
        )
        log.info("%-22s %d datasets", name, len(refs))

    print()
    print("Run the task with:")
    print()
    print("  pipetask run \\")
    print(f"    -b {repo} \\")
    print(f"    -i {DEFAULTS_CHAIN} \\")
    print("    -o u/$USER/blitz \\")
    print("    -t lsst.ts.wep.blitz.donutBlitzCorner.DonutBlitzCornerTask \\")
    if visits:
        print(f'    -d "instrument=\'{INSTRUMENT}\' AND visit={sorted(v.id for v in visits)[0]}"')
    print()


if __name__ == "__main__":
    sys.exit(main())
