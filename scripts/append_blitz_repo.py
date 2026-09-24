#!/usr/bin/env python
"""Append a new export_blitz_repo.py export into an existing local repo.

Run this on the laptop, on a *new* directory rsynced down from USDF (produced
by a fresh `export_blitz_repo.py --visit ... --output ...` run, since the
export script refuses to write into an existing output directory). Unlike
build_blitz_repo.py this does not create the repo or register the instrument
or curated calibrations -- those already exist from the first build, and
`butler create` / `register-instrument` are not safe to repeat.

Steps, in order:

1. `butler import` -- the new datasets and their dimension records.
   Calibration collections come back certified the same way the initial
   import did.
2. `define-visits` -- safe to rerun: it only defines visits that are not
   already defined, so the existing visit is untouched.
3. `collection-chain LSSTCam/defaults` -- redefined from scratch, same as
   build_blitz_repo.py, since the new import creates a new RUN collection
   that must be added to the chain.

If the new visit resolves to calibs already imported under the *same*
CALIBRATION collection name, `butler import` fails hard on the dataset ingest
(`UNIQUE constraint failed: dataset_location...`) rather than silently
skipping the duplicate -- confirmed on a throwaway /tmp repo with two exports
that shared one certified calib dataset. `_dedup_export()` below strips any
dataset (and its associations) whose UUID is already present in the
destination before `butler import` ever sees it, since the export always
carries the full resolved calib/refcat set per run rather than just what is
new.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

from build_blitz_repo import DEFAULTS_CHAIN, INSTRUMENT_CLASS, discover_children, report, run

log = logging.getLogger("append_blitz_repo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export_dir", help="New directory rsynced from USDF (holds export.yaml).")
    parser.add_argument("repo", help="Path of the existing repo to append into.")
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


def _existing_dataset_ids(repo: str) -> set:
    """UUIDs of every dataset already in the destination repo.

    Used to drop already-imported datasets from a new export before
    `butler import` sees them, since a duplicate dataset_id fails the
    datastore ingest rather than being skipped.
    """
    from lsst.daf.butler import Butler

    butler = Butler.from_config(repo)
    ids = set()
    for dt in butler.registry.queryDatasetTypes("*"):
        for ref in butler.query_datasets(
            dt.name, collections="*", find_first=False, explain=False, limit=None
        ):
            ids.add(ref.id)
    return ids


def _dedup_export(export_yaml: str, existing_ids: set) -> str:
    """Write a copy of `export_yaml` with already-imported datasets removed.

    Drops whole `dataset` records whose UUID is in `existing_ids`, and strips
    those same UUIDs out of `associations` blocks (both TAGGED `dataset_ids`
    and CALIBRATION `validity_ranges[].dataset_ids`). A validity range left
    with no dataset_ids is dropped entirely, and a `dataset` record whose
    every UUID is already present is dropped -- both are already in the
    destination, so re-saving them would either no-op or conflict.

    Returns the path to the deduplicated copy, written alongside the
    original as `export.deduped.yaml`.
    """
    with open(export_yaml) as f:
        doc = yaml.safe_load(f)

    dropped = 0
    kept_data = []
    for entry in doc["data"]:
        if entry["type"] == "dataset":
            kept_records = []
            for record in entry["records"]:
                if all(i in existing_ids for i in record["dataset_id"]):
                    dropped += len(record["dataset_id"])
                    continue
                kept_records.append(record)
            if kept_records:
                entry["records"] = kept_records
                kept_data.append(entry)
        elif entry["type"] == "associations":
            if "dataset_ids" in entry:
                entry["dataset_ids"] = [i for i in entry["dataset_ids"] if i not in existing_ids]
                if entry["dataset_ids"]:
                    kept_data.append(entry)
            else:
                ranges = []
                for r in entry["validity_ranges"]:
                    r["dataset_ids"] = [i for i in r["dataset_ids"] if i not in existing_ids]
                    if r["dataset_ids"]:
                        ranges.append(r)
                if ranges:
                    entry["validity_ranges"] = ranges
                    kept_data.append(entry)
        else:
            kept_data.append(entry)

    log.info("Deduped export: dropped %d dataset(s) already in the destination", dropped)
    doc["data"] = kept_data

    deduped_path = os.path.join(os.path.dirname(export_yaml), "export.deduped.yaml")
    with open(deduped_path, "w") as f:
        yaml.dump(doc, f, sort_keys=False)
    return deduped_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    export_yaml = os.path.join(args.export_dir, "export.yaml")
    if not os.path.exists(export_yaml):
        log.error("No export.yaml in %s -- is that the rsynced export directory?", args.export_dir)
        return 1
    if not os.path.exists(args.repo):
        log.error("Repo does not exist: %s -- use build_blitz_repo.py for a first import.", args.repo)
        return 1

    keep = args.keep_going

    existing_ids = _existing_dataset_ids(args.repo)
    log.info("Destination already has %d datasets", len(existing_ids))
    deduped_yaml = _dedup_export(export_yaml, existing_ids)

    run(
        [
            "butler",
            "import",
            args.repo,
            args.export_dir,
            "--export-file",
            deduped_yaml,
            "--transfer",
            args.transfer,
        ],
        keep_going=keep,
    )
    run(["butler", "define-visits", args.repo, INSTRUMENT_CLASS], keep_going=keep)

    children = discover_children(args.repo)
    log.info("Chaining into %s: %s", DEFAULTS_CHAIN, children)
    if children:
        run(
            ["butler", "collection-chain", args.repo, DEFAULTS_CHAIN, "--mode", "redefine", *children],
            keep_going=keep,
        )

    log.info("Appended export from %s into %s", args.export_dir, args.repo)
    report(args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
