#!/usr/bin/env python
"""A/B the trigger's `encode` step: serial vs. parallel, and split the serial path.

`client.run_once` reports encode as one number ("encode -> ... 1.347s"). This
takes it apart: the serial path's load/encode/pack split per detector, then the
same exposure through `daq.encoded_blob`, which forks a worker per sensor
writing pixels straight into shared memory.

Run this on the machine you care about. The ratios travel between machines; the
absolute numbers do not.

Each timed pass uses a *different* exposure directory, since re-reading one
measures the page cache rather than the work. Pass --warm to also re-run on an
already-read directory.

    scripts/bench_encode.py $DONUT_SERVER_REPLAY_DIR
    scripts/bench_encode.py $DONUT_SERVER_REPLAY_DIR --warm --workers 2 4 8
"""
import argparse
import glob
import os
import time

# Before anything reaches numpy, so the parallel path can actually fork -- the
# same block `bin.src/donutTrigger.py` has, for the same reason. Without it
# `daq.encoded_blob` declines and silently measures the serial fallback twice.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from lsst.ts.donut_server import client, daq, exposure_codec, protocol  # noqa: E402


def one_detector(handle):
    """What the serial `build_raw_parts` does per detector, timed in halves."""
    t0 = time.perf_counter()
    exposure = daq.load_raw(handle)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    name = exposure.getDetector().getName()
    img_blob, meta_blob = exposure_codec.encode_exposure(exposure)
    t_encode = time.perf_counter() - t0

    return name, img_blob, meta_blob, t_load, t_encode


def serial_pass(source):
    per_det = []
    parts = {}
    t_wall = time.perf_counter()
    for det_id in sorted(source.handles):
        name, img, meta, t_load, t_enc = one_detector(source.handles[det_id])
        parts[protocol.part_name(name, protocol.IMG_SUFFIX)] = img
        parts[protocol.part_name(name, protocol.META_SUFFIX)] = meta
        per_det.append((name, t_load, t_enc))
    wall = time.perf_counter() - t_wall

    t0 = time.perf_counter()
    blob = protocol.pack_blob(parts)
    t_pack = time.perf_counter() - t0
    return per_det, wall, t_pack, len(blob)


def parallel_pass(source, n_workers):
    """`daq.encoded_blob` end to end, including block setup and teardown."""
    t0 = time.perf_counter()
    with daq.encoded_blob(source, n_workers) as view:
        total = len(view)
        elapsed = time.perf_counter() - t0
    return elapsed, total


def report_serial(label, per_det, wall, t_pack, nbytes):
    load = sum(t for _, t, _ in per_det)
    enc = sum(t for _, _, t in per_det)
    print(f"\n=== {label} ===")
    print(f"{'detector':>10}  {'load_s':>8}  {'encode_s':>9}")
    for name, t_load, t_enc in per_det:
        print(f"{name:>10}  {t_load:8.3f}  {t_enc:9.3f}")
    print(f"{'sum':>10}  {load:8.3f}  {enc:9.3f}")
    print(
        f"  loop wall: {wall:.3f}s   pack_blob: {t_pack:.3f}s   "
        f"total: {wall + t_pack:.3f}s   blob: {nbytes} bytes"
    )
    print(
        f"  load {load / (load + enc) * 100:.0f}% / encode "
        f"{enc / (load + enc) * 100:.0f}% of per-detector work"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "replay_dir",
        nargs="?",
        default=os.environ.get("DONUT_SERVER_REPLAY_DIR"),
        help="directory holding the exposure subdirs (default: "
        "$DONUT_SERVER_REPLAY_DIR)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        nargs="*",
        default=[2, 4, 8],
        help="worker counts to test for the parallel path (default: 2 4 8)",
    )
    ap.add_argument(
        "--warm",
        action="store_true",
        help="also re-run the serial pass on the already-read exposure",
    )
    args = ap.parse_args()
    if not args.replay_dir:
        ap.error("no replay_dir given and $DONUT_SERVER_REPLAY_DIR is unset")

    dirs = sorted(
        d for d in glob.glob(os.path.join(args.replay_dir, "*")) if os.path.isdir(d)
    )
    needed = 2 + len(args.workers)
    if len(dirs) < needed:
        ap.error(f"need {needed} exposure dirs for cold passes, found {len(dirs)}")

    try:
        daq._assert_single_threaded_blas()
    except RuntimeError as exc:
        ap.error(f"{exc}\nThe parallel numbers would be the serial fallback.")

    # Warmed on a directory then discarded, so the first timed pass is not also
    # paying for LsstCam.getCamera() and the lazy afw imports.
    warmup = dirs.pop(0)
    source, _ = daq.resolve_exposure_dir(warmup)
    one_detector(source.handles[sorted(source.handles)[0]])
    print(f"warmed camera + imports on {os.path.basename(warmup)}")

    cold = dirs.pop(0)
    source, _ = daq.resolve_exposure_dir(cold)
    per_det, wall, t_pack, nbytes = serial_pass(source)
    report_serial(
        f"serial, cold: {os.path.basename(cold)}", per_det, wall, t_pack, nbytes
    )
    serial_total = wall + t_pack

    if args.warm:
        source, _ = daq.resolve_exposure_dir(cold)
        per_det, wall, t_pack, nbytes = serial_pass(source)
        report_serial(
            f"serial, warm (re-read): {os.path.basename(cold)}",
            per_det,
            wall,
            t_pack,
            nbytes,
        )

    print("\n=== parallel, each on a fresh cold exposure ===")
    print(f"{'workers':>8}  {'total_s':>9}  {'speedup':>8}  {'bytes':>10}")
    print(f"{'serial':>8}  {serial_total:9.3f}  {1.0:8.2f}  {nbytes:10}")
    for workers in args.workers:
        directory = dirs.pop(0)
        source, _ = daq.resolve_exposure_dir(directory)
        elapsed, total = parallel_pass(source, workers)
        print(
            f"{workers:>8}  {elapsed:9.3f}  {serial_total / elapsed:8.2f}  "
            f"{total:10}   ({os.path.basename(directory)})"
        )

    print(
        "\nThe parallel column includes creating, writing, and unlinking the "
        "shared block -- i.e. everything `encode ->` reports."
    )


if __name__ == "__main__":
    main()
