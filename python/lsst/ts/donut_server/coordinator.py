"""The long-lived engine that runs the real wavefront pipeline.

Holds the module-globals `_CALIB_STORE` and `_REFCAT_STORE`, populated at
`prepare` time and reused across jobs (each skipped on repeat if its own key --
band + calib selector, or pointing -- is unchanged). Per job,
`run_job` rebuilds the raw exposures out of shared memory, hands them to a
hand-built in-memory Butler, and calls the real
`DonutBlitzCornerTask.runQuantum()`.

The task does its own two-stage forking internally (cutout, then WF fit), sized
by `ExecutionResources(num_cores=...)`, and uses the same
populate-a-module-global-before-forking trick this service was prototyped
around -- so there are no worker pools here any more. What remains here is the
process and memory discipline that forking safely depends on.

This module is designed to run as a *separate* process, started by the
FastAPI front-end via `multiprocessing.get_context("spawn")` -- never fork,
since the front-end is async/threaded and fork-from-threaded-process is
unsafe. The coordinator itself stays single-threaded (enforced by the assert
in `coordinator_main`) so that the task's forks are safe.
"""
from __future__ import annotations

import os

# Must happen before numpy import: keep BLAS/OpenMP single-threaded so 8
# fork workers don't each fork with a live thread pool. run_server.sh exports
# the same vars, which is what makes this order-independent in the real service;
# this block is the fallback for anything that imports coordinator directly.
# Verified at startup by _assert_single_threaded_blas.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import ctypes
import gc
import glob
import io
import logging
import multiprocessing as mp
import re
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, NamedTuple

import pyarrow.parquet

import lsst.afw.image as afwImage
import lsst.ip.isr as ipIsr

# Imported at module scope, never lazily inside a worker: every one of these
# must be fully resident before the first fork.
from lsst.ts.donut_server import protocol
from lsst.ts.donut_server import exposure_codec
from lsst.ts.donut_server import logtail
from lsst.ts.donut_server import refcat_store
from lsst.daf.butler import (
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionUniverse,
    Quantum,
)
from lsst.daf.butler.formatters.parquet import astropy_to_arrow
from lsst.pipe.base import QuantumContext
from lsst.pipe.base._quantumContext import ExecutionResources
from lsst.pipe.base.tests.in_memory_limited_butler import InMemoryLimitedButler
from lsst.ts.wep.blitz.donutBlitzCorner import (
    DonutBlitzCornerConfig,
    DonutBlitzCornerTask,
)

_log = logging.getLogger(__name__)

NUM_WORKERS = 8

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calib")

# Per-donut postage stamps: 18.5 + 4.6 + 4.6 MB of the result table's 28 MB.
# Dropped before the result leaves this process -- the remaining 58 columns,
# including all the Zernikes, are 247 KB as parquet.
IMAGE_COLUMNS = ("stamp", "wf_img", "model_img")

# Nothing here validates the instrument against a real Registry, and the blitz
# task never uses it functionally (its real visit id comes off the raw header).
INSTRUMENT = "LSSTCam"

# Arbitrary; no real collection is involved.
RUN = "donut_server"

# Populated by load_calibs_for_prepare; the objects here are handed to the
# in-memory Butler, and the task's own fork workers inherit them via CoW.
_CALIB_STORE: dict[str, Any] = {}

# Populated at prepare time from the boresight alone, before any pixels exist --
# which is the whole reason prepare takes a boresight, since resharding the local
# level-5 files to the level-7 granularity the loader wants costs ~450 ms that
# would otherwise land on the push critical path. Held outside _CALIB_STORE
# because it turns over on a different key: pointing, not band + calib selector.
_REFCAT_STORE = refcat_store.RefCatStore()

# The reusable shared block the front-end streams raw pixels into, opened once
# in coordinator_main and read in place -- raw pixels never cross the Pipe.
_SHM: Any = None
_SHM_VIEW: Any = None

# Band-independent and butler-independent, so these survive calib reloads and
# are deliberately not inside _CALIB_STORE (which gets cleared on reload).
_UNIVERSE: Any = None
_TASK: Any = None


class QuantumBundle(NamedTuple):
    """Everything runQuantum needs, plus the handles to read its output back."""

    butler_qc: Any
    input_refs: Any
    output_refs: Any
    butler: Any
    output_ref: Any
    timings: dict


def _universe_and_task() -> tuple[Any, Any]:
    """Build (once) the DimensionUniverse and the blitz task instance.

    Neither needs a Registry or a repo config, and task construction never
    touches a Butler (its subtasks are config-driven), so both can be made
    once and reused across jobs.
    """
    global _UNIVERSE, _TASK
    if _UNIVERSE is None:
        _UNIVERSE = DimensionUniverse()
        _TASK = DonutBlitzCornerTask(config=DonutBlitzCornerConfig())
    return _UNIVERSE, _TASK


def _warm_up_parquet() -> None:
    """First astropy_to_arrow() call costs ~400 ms of pyarrow init; the rest are
    ~23 ms. Pay it at prepare time, which is where raw-independent precompute
    belongs, rather than on the first push."""
    from astropy.table import Table

    astropy_to_arrow(Table({"a": [[0.0, 1.0]]}))


def to_parquet(table) -> bytes:
    """Serialize a result table the same way a real Butler would.

    Uses daf_butler's own ArrowAstropy conversion, so the bytes match what
    `butler.put` of a `donutBlitzCornerResults` dataset writes -- including the
    multidimensional Zernike columns and per-column units.
    """
    buf = io.BytesIO()
    pyarrow.parquet.write_table(astropy_to_arrow(table), buf)
    return buf.getvalue()


@dataclass
class CalibSet:
    """Real calibrations for prepare's band, keyed by detector name.

    intrinsic_zernikes_by_name is partial: a detector missing that file
    simply has no entry (mirrors the real Butler connection's minimum=0).
    """

    config_key: tuple
    detector_ids: list
    ptc_by_name: dict
    linearizer_by_name: dict
    crosstalk_by_name: dict
    flat_by_name: dict
    intrinsic_zernikes_by_name: dict


def _discover_detector_ids(calib_dir: str) -> list:
    """Authoritative detector-id list: every id with a ptc_*.fits file."""
    ids = []
    for path in sorted(glob.glob(os.path.join(calib_dir, "ptc_*.fits"))):
        m = re.fullmatch(r"ptc_(\d+)\.fits", os.path.basename(path))
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise RuntimeError(f"No ptc_*.fits calib files found in {calib_dir!r}")
    return ids


def load_calibs_for_prepare(band: str, calib_selector: str) -> dict:
    """Populate _CALIB_STORE with real calibs for `band`. Skips reload if
    config is unchanged (reuse guard)."""
    config_key = (band, calib_selector)
    t0 = time.monotonic()

    # Pay the DimensionUniverse + task construction + pyarrow init here, not on push.
    _universe_and_task()
    _warm_up_parquet()

    if _CALIB_STORE.get("config_key") == config_key:
        return {"reused": True, "elapsed_s": time.monotonic() - t0}

    detector_ids = _discover_detector_ids(CALIB_DIR)

    ptc_by_name: dict = {}
    linearizer_by_name: dict = {}
    crosstalk_by_name: dict = {}
    flat_by_name: dict = {}
    intrinsic_zernikes_by_name: dict = {}

    for det_id in detector_ids:
        # Band-independent, required. Detector name comes off the calib
        # object itself (._detectorName) -- no separate id->name table needed.
        ptc = ipIsr.PhotonTransferCurveDataset.readFits(
            os.path.join(CALIB_DIR, f"ptc_{det_id}.fits")
        )
        name = ptc._detectorName
        ptc_by_name[name] = ptc
        linearizer_by_name[name] = ipIsr.Linearizer.readFits(
            os.path.join(CALIB_DIR, f"linearizer_{det_id}.fits")
        )
        crosstalk_by_name[name] = ipIsr.CrosstalkCalib.readFits(
            os.path.join(CALIB_DIR, f"crosstalk_{det_id}.fits")
        )

        # Band-dependent, required.
        flat_path = os.path.join(CALIB_DIR, f"flat_{det_id}_{band}.fits")
        if not os.path.exists(flat_path):
            raise RuntimeError(
                f"Missing required flat calib for detector {name} ({det_id}), "
                f"band {band!r}: {flat_path}"
            )
        flat_by_name[name] = afwImage.ExposureF.readFits(flat_path)

        # Band-dependent, optional (mirrors the real Butler connection's
        # minimum=0): missing file just means no entry for this detector.
        iz_path = os.path.join(CALIB_DIR, f"intrinsicZernikes_{det_id}_{band}.fits")
        if os.path.exists(iz_path):
            intrinsic_zernikes_by_name[name] = ipIsr.IsrCalib.readFits(iz_path)

    calib = CalibSet(
        config_key=config_key,
        detector_ids=detector_ids,
        ptc_by_name=ptc_by_name,
        linearizer_by_name=linearizer_by_name,
        crosstalk_by_name=crosstalk_by_name,
        flat_by_name=flat_by_name,
        intrinsic_zernikes_by_name=intrinsic_zernikes_by_name,
    )
    _CALIB_STORE.clear()
    _CALIB_STORE["config_key"] = config_key
    _CALIB_STORE["calib"] = calib

    return {
        "reused": False,
        "elapsed_s": time.monotonic() - t0,
        "n_detectors": len(detector_ids),
        "n_intrinsic_zernikes": len(intrinsic_zernikes_by_name),
        "band": band,
    }


def reconstruct_exposures(layout: list) -> dict[str, Any]:
    """Rebuild one ExposureF per sensor, reading pixels in place out of the
    shared block. Keyed by detector name."""
    parts = {
        part.name: _SHM_VIEW[part.offset : part.offset + part.byte_count]
        for part in layout
    }
    try:
        return {
            sensor: exposure_codec.decode_exposure(img_blob, meta_blob)
            for sensor, (img_blob, meta_blob) in protocol.split_parts(parts).items()
        }
    finally:
        # Sub-views keep the block's buffer exported; release them so the block
        # can be closed cleanly at shutdown.
        for sub_view in parts.values():
            sub_view.release()


def build_quantum_context(exposures: dict[str, Any], calib: CalibSet) -> QuantumBundle:
    """Build the InMemoryLimitedButler / Quantum / in+out refs for one exposure.

    No Registry, SQLite or obs_lsst camera package is involved: the dimension
    universe, dataset types, data coordinates and quantum are all hand-built.
    The result is ready to hand straight to
    `DonutBlitzCornerTask.runQuantum()`.
    """
    universe, task = _universe_and_task()
    conns = task.config.connections.ConnectionsClass(config=task.config)
    input_names = list(conns.inputs) + list(conns.prerequisiteInputs)
    out_name = "cornerResults"

    dtypes = {}
    for name in input_names + [out_name]:
        conn = getattr(conns, name)
        dtypes[name] = DatasetType(
            conn.name,
            tuple(conn.dimensions),
            conn.storageClass,
            universe=universe,
            isCalibration=getattr(conn, "isCalibration", False),
        )

    missing = sorted(set(exposures) - set(calib.ptc_by_name))
    if missing:
        raise RuntimeError(f"no calibs loaded for detector(s): {missing}")

    visit_ids = {exp.getInfo().getVisitInfo().id for exp in exposures.values()}
    if len(visit_ids) != 1:
        raise RuntimeError(f"raws span multiple visits: {sorted(visit_ids)}")
    exposure_id = visit_ids.pop()

    sample = next(iter(exposures.values()))
    physical_filter = sample.getFilter().physicalLabel
    band = sample.getFilter().bandLabel
    metadata = sample.getMetadata()
    day_obs = int(metadata["DAYOBS"])
    group = str(metadata["GROUPID"])

    def data_id(name: str, **values):
        # Every key in the dataset type's dimensions must be supplied, implied
        # ones included -- `band` is implied by `physical_filter`, and omitting
        # it silently yields hasFull() == False, which later makes the task's
        # _exposure_group() blow up on dataId["group"].
        return DataCoordinate.standardize(values, dimensions=dtypes[name].dimensions)

    refs: dict[str, list] = {name: [] for name in input_names}
    payload: dict[Any, Any] = {}

    # Exact coverage check rather than an angular tolerance on the boresight: a
    # prepare/push pointing mismatch would otherwise degrade astrometry silently
    # instead of failing, the same failure mode the band cross-check guards.
    uncovered = _REFCAT_STORE.uncovered(refcat_store.shard_ids_for_exposures(exposures))
    if uncovered:
        raise RuntimeError(
            f"refcat shards loaded at prepare do not cover these raws: missing "
            f"level-{refcat_store.SHARD_LEVEL} shard(s) {sorted(uncovered)}. The "
            "boresight given to prepare does not match the raws."
        )

    for htm_index, catalog in _REFCAT_STORE.shards.items():
        ref = DatasetRef(
            dtypes["refCat"],
            refcat_store.htm_data_id(universe, htm_index),
            run=RUN,
        )
        refs["refCat"].append(ref)
        payload[ref] = catalog

    for det_name in sorted(exposures):
        exp = exposures[det_name]
        det_id = exp.getDetector().getId()
        exposure_did = data_id(
            "raws", instrument=INSTRUMENT, exposure=exposure_id, detector=det_id,
            day_obs=day_obs, group=group, physical_filter=physical_filter, band=band,
        )
        detector_did = data_id("ptc", instrument=INSTRUMENT, detector=det_id)
        filtered_did = data_id(
            "flat", instrument=INSTRUMENT, detector=det_id,
            physical_filter=physical_filter, band=band,
        )

        entries = [
            ("raws", exposure_did, exp),
            ("ptc", detector_did, calib.ptc_by_name[det_name]),
            ("linearizer", detector_did, calib.linearizer_by_name[det_name]),
            ("crosstalk", detector_did, calib.crosstalk_by_name[det_name]),
            ("flat", filtered_did, calib.flat_by_name[det_name]),
        ]
        # minimum=0 upstream: a detector with no intrinsicZernikes simply
        # contributes no ref.
        if det_name in calib.intrinsic_zernikes_by_name:
            entries.append(
                ("intrinsicZernikes", filtered_did, calib.intrinsic_zernikes_by_name[det_name])
            )

        for name, did, obj in entries:
            ref = DatasetRef(dtypes[name], did, run=RUN)
            refs[name].append(ref)
            payload[ref] = obj

    visit_did = data_id(
        out_name, instrument=INSTRUMENT, visit=exposure_id, day_obs=day_obs,
        physical_filter=physical_filter, band=band,
    )
    out_ref = DatasetRef(dtypes[out_name], visit_did, run=RUN)

    # put() is a bare dict insert, so this is ~1 ms for the whole input set and
    # shares the CalibSet objects rather than copying them. The deep copy lands
    # on get() instead (copy=True is hardcoded in InMemoryLimitedButler.put),
    # which means a fork worker calling butlerQC.get() materializes its own
    # ~100 MB copy -- CoW buys nothing on that path.
    butler = InMemoryLimitedButler(universe, list(dtypes.values()))
    t0 = time.perf_counter()
    for ref, obj in payload.items():
        butler.put(obj, ref)
    put_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    quantum = Quantum(
        taskName=type(task).__name__,
        dataId=visit_did,
        # Every input/prerequisiteInput needs a key, present even when its list
        # is empty: omitting one makes buildDatasetRefs raise KeyError on its
        # dataset type name.
        inputs={dtypes[name]: refs[name] for name in input_names},
        outputs={dtypes[out_name]: [out_ref]},
    )
    input_refs, output_refs = conns.buildDatasetRefs(quantum)
    refs_s = time.perf_counter() - t0

    butler_qc = QuantumContext(
        butler, quantum, resources=ExecutionResources(num_cores=NUM_WORKERS)
    )

    timings = {
        "butler_put_s": put_s,
        "dataset_refs_s": refs_s,
        "quantum": {
            "visit": exposure_id,
            "exposure": exposure_id,
            "band": band,
            "physical_filter": physical_filter,
            "group": group,
            "n_raws": len(refs["raws"]),
            "n_refcat_shards": len(refs["refCat"]),
            "n_input_datasets": len(payload),
        },
    }
    return QuantumBundle(
        butler_qc=butler_qc,
        input_refs=input_refs,
        output_refs=output_refs,
        butler=butler,
        output_ref=out_ref,
        timings=timings,
    )


def _summarize(table) -> dict:
    """Small JSON-safe digest of the result table, for /status and /result."""
    det_names = [str(name) for name in table["det_name"]]
    per_detector: dict[str, int] = {}
    for name in det_names:
        per_detector[name] = per_detector.get(name, 0) + 1

    groups: dict[Any, bool] = {}
    for group_id, success in zip(table["group_id"], table["group_fit_success"]):
        groups[str(group_id)] = bool(success)

    return {
        "n_rows": len(table),
        "n_detectors": len(per_detector),
        "rows_per_detector": dict(sorted(per_detector.items())),
        "n_groups": len(groups),
        "n_groups_succeeded": sum(groups.values()),
        "columns": list(table.colnames),
    }


def run_job(job_id: str, layout: list) -> dict:
    """Rebuild the raws, build the butler, run the real task."""
    t_total = time.perf_counter()
    # The only delimiter between one job's log lines and the next's, since the
    # task's own logging knows nothing about jobs.
    _log.info("job %s: starting", job_id)

    # Reads straight out of the shared block, so there is no second copy of the
    # ~308 MB payload to free afterwards -- the exposures themselves (~900 MB,
    # since ExposureF allocates all three planes) are the only per-job bulk.
    t0 = time.perf_counter()
    exposures = reconstruct_exposures(layout)
    decode_s = time.perf_counter() - t0

    band = next(iter(exposures.values())).getFilter().bandLabel
    prepared = _CALIB_STORE.get("config_key")
    prepared_band = prepared[0] if prepared else None
    if prepared_band != band:
        raise RuntimeError(
            f"raws are band {band!r} but prepare loaded band {prepared_band!r} "
            "-- the flats would be wrong"
        )
    calib = _CALIB_STORE["calib"]

    bundle = build_quantum_context(exposures, calib)
    _, task = _universe_and_task()

    # The task forks its own cutout and WF-fit pools internally.
    #
    # There is deliberately no gc.freeze() here. It was standard practice in
    # this service's prototype, but measured against the real object graph it
    # changes neither CoW faults nor wall time (both within run-to-run noise):
    # freeze only stops the *collector* writing to gc headers, and these
    # short-lived numeric workers may never trigger a collection, while the
    # refcount writes that actually dirty pages are unaffected by it. It is also
    # actively dangerous -- frozen objects are permanently exempt from cyclic
    # collection, so a per-job freeze without a matching unfreeze stranded ~1,100
    # objects every job. Not worth keeping for no measured gain.
    t0 = time.perf_counter()
    task.runQuantum(bundle.butler_qc, bundle.input_refs, bundle.output_refs)
    run_quantum_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_table = bundle.butler.get(bundle.output_ref)
    keep = [name for name in full_table.colnames if name not in IMAGE_COLUMNS]
    table = full_table[keep]
    parquet_bytes = to_parquet(table)
    serialize_s = time.perf_counter() - t0

    # Summarize the slimmed table, so `columns` describes exactly what the
    # client receives.
    summary = _summarize(table)
    summary["dropped_columns"] = list(IMAGE_COLUMNS)
    summary["parquet_bytes"] = len(parquet_bytes)

    bundle_timings = bundle.timings
    quantum_info = bundle_timings.pop("quantum")

    # Reclaim this job's cycles now rather than whenever the automatic collector
    # next fires: the ~900 MB of exposures should be gone well before the next
    # exposure arrives (~30 s cadence).
    del bundle, full_table, table, exposures
    t0 = time.perf_counter()
    collected = gc.collect()
    gc_s = time.perf_counter() - t0

    return {
        "summary": summary,
        "quantum": quantum_info,
        "table_parquet": parquet_bytes,
        "timings": {
            "decode_s": decode_s,
            **bundle_timings,
            "run_quantum_s": run_quantum_s,
            "serialize_s": serialize_s,
            "gc_s": gc_s,
            "gc_collected": collected,
            "coordinator_total_s": time.perf_counter() - t_total,
        },
    }


def _assert_single_threaded_blas() -> None:
    """Check that the env block at the top of this module actually bound.

    It only binds if this module is imported before numpy, and the failure is
    silent: measured on an M3 Pro, an importer that loads numpy first leaves
    OpenBLAS at 12 threads, which becomes 12 threads in every one of the 8 fork
    workers. That costs no wall time (A/B'd: dead heat), but this numpy links
    OpenBLAS built with USE_OPENMP, and forking with live OpenMP threads is
    undefined behaviour -- so fail at startup rather than fork into it.

    Both symbols are resolvable via the global handle because libopenblas and
    libomp are already loaded by this module's own imports.
    """
    lib = ctypes.CDLL(None)
    hot = {
        name: n
        for name, n in (
            ("OpenBLAS", lib.openblas_get_num_threads()),
            ("OpenMP", lib.omp_get_max_threads()),
        )
        if n != 1
    }
    if hot:
        raise RuntimeError(
            f"thread pools not clamped: {hot}. Something imported numpy before "
            "coordinator, so the thread-limit env vars never bound. Export them "
            "before starting the process (see run_server.sh)."
        )


def configure_logging() -> str:
    """Send the pipeline's log stream to a file, and return its path.

    Nothing else in this service configures logging, which is not a neutral
    default: with no handler anywhere, the stack's loggers fall back to
    `logging.lastResort` -- stderr, WARNING and above -- so every
    `self.log.info` the blitz task emits is discarded. That is the entire
    running commentary on a job (per-detector cutout timings, the WF dispatch
    line, the per-detector WCS refit summaries), and losing it leaves warnings
    and tracebacks as the only evidence a job ran at all.

    Three details here are load-bearing rather than cosmetic:

    - Level is set on the root logger, not on one named tree. The task's own
      `self.log` is named for the task (`donutBlitzCorner...`), while the
      helper modules use `logging.getLogger(__name__)` under `lsst.ts.wep.*`;
      the interesting lines span both namespaces.
    - `%(process)d` is in the format because the task forks 8 cutout and WF
      workers which inherit this handler across the fork and write to the same
      file. Without the pid, interleaved lines cannot be attributed.
    - The handler is attached before any fork, and `StreamHandler.emit`
      flushes after every record. Between records the handler's buffer is
      therefore empty, so a child forked between two records inherits nothing
      to flush twice, and each record reaches the append-mode fd as a single
      atomic write instead of interleaving mid-line with a sibling's.
    """
    path = logtail.log_path()
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s pid=%(process)-6d %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return path


def coordinator_main(conn, shm_name: str) -> None:
    """Serial command loop: recv one command, dispatch, send response, repeat."""
    # Become our own process-group leader so the front-end can os.killpg() this
    # process *and* the task's fork workers. Reaping only the coordinator would
    # leave 8 orphans holding the shared block mapping and 8 cores.
    #
    # Side effect: this process no longer receives the terminal's SIGINT. Ctrl-C
    # still shuts it down, because uvicorn's handler runs lifespan shutdown ->
    # coord.aclose(); and front-end death still closes the last parent_conn copy,
    # so the recv() below raises EOFError and this process exits.
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    assert threading.active_count() == 1, "coordinator must be single-threaded at fork time"
    _assert_single_threaded_blas()

    global _SHM, _SHM_VIEW
    # track=False: the front-end created this block and is the only process that
    # should unlink it. Without it, this process's resource_tracker would unlink
    # the block when the process exits.
    _SHM = shared_memory.SharedMemory(name=shm_name, track=False)
    _SHM_VIEW = memoryview(_SHM.buf)

    # Sent only once, and only after the BLAS assert and the shared block are
    # both good -- so "hello" means fully ready to serve, not merely spawned.
    # Without it the parent cannot distinguish a healthy child from one that is
    # 15 s into importing afw + ts_wep, or one that is about to fail the assert.
    conn.send({"ok": True, "event": "hello", "pid": os.getpid()})

    # After the hello, deliberately. Everything above -- the single-thread
    # assert, the BLAS clamp check, opening the shared block -- fails by
    # propagating out of this function, and its traceback is only useful on the
    # stderr this process inherited from whoever started the server. Configuring
    # the file handler earlier would redirect those startup failures into a file
    # that nobody is tailing yet, since the path is only announced at startup.
    log_path = configure_logging()
    _log.info("coordinator ready, pid %d, logging to %s", os.getpid(), log_path)

    try:
        while True:
            try:
                command = conn.recv()
            except EOFError:
                break

            cmd = command.get("cmd")
            if cmd == "shutdown":
                conn.send({"ok": True})
                break
            elif cmd == "prepare":
                try:
                    timings = {
                        "calib": load_calibs_for_prepare(
                            command["band"], command["calib_selector"]
                        ),
                        "refcat": _REFCAT_STORE.ensure(
                            command["boresight_ra"], command["boresight_dec"]
                        ),
                    }
                    conn.send({"ok": True, "timings": timings})
                except Exception as exc:
                    conn.send({"ok": False, "error": str(exc)})
            elif cmd == "push":
                try:
                    result = run_job(command["job_id"], command["layout"])
                    conn.send({"ok": True, "result": result})
                except Exception as exc:
                    conn.send({"ok": False, "error": str(exc)})
            else:
                conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
    finally:
        _SHM_VIEW.release()
        _SHM_VIEW = None
        _SHM.close()
        _SHM = None


if __name__ == "__main__":
    # Standalone smoke test: drives the coordinator through prepare -> push
    # with real raws and no FastAPI involved, to validate fork/CoW mechanics and
    # the butler build in isolation.
    from lsst.ts.donut_server import client

    VISIT = 2026071300478  # the r-band exposure
    source = client.resolve_from_files(client.RAW_DIR, VISIT)
    band = source.band
    print(
        f"exposure -> visit={source.visit} band={band} "
        f"detectors={sorted(source.handles)}"
    )

    boresight_ra, boresight_dec = client.read_boresight(source)
    print(f"boresight -> ra={boresight_ra:.4f} dec={boresight_dec:.4f}")

    t0 = time.monotonic()
    blob = protocol.pack_blob(client.build_raw_parts(source))
    print(f"encode -> {len(blob)} bytes ({time.monotonic() - t0:.3f}s)")

    # Stand in for the front-end: own the shared block, stream the blob in, and
    # hand the coordinator only the layout.
    shm = shared_memory.SharedMemory(create=True, size=len(blob))
    view = memoryview(shm.buf)
    view[: len(blob)] = blob
    layout = protocol.parse_layout(view[: len(blob)])

    spawn_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp.Pipe()
    proc = spawn_ctx.Process(target=coordinator_main, args=(child_conn, shm.name))
    proc.start()
    # The parent must drop its copy of the child end, or the socketpair can never
    # reach EOF and a recv() after the child dies would block forever instead of
    # raising. Same reason as in server.py's Coord.start().
    child_conn.close()

    # coordinator_main sends this once when it is ready to serve. Consume it here
    # or every response below is off by one.
    print("hello ->", parent_conn.recv())

    parent_conn.send({
        "cmd": "prepare",
        "band": band,
        "calib_selector": "default",
        "boresight_ra": boresight_ra,
        "boresight_dec": boresight_dec,
    })
    print("prepare ->", parent_conn.recv())

    t0 = time.monotonic()
    parent_conn.send({"cmd": "push", "job_id": "smoke-1", "layout": layout})
    resp = parent_conn.recv()
    elapsed = time.monotonic() - t0

    print("push ok:", resp.get("ok"), f"elapsed: {elapsed:.3f}s")
    if resp.get("ok"):
        result = resp["result"]
        print("quantum:", result["quantum"])
        print("timings:", result["timings"])
        summary = dict(result["summary"])
        summary.pop("columns")
        print("summary:", summary)

        from lsst.daf.butler.formatters.parquet import arrow_to_astropy

        blitz = arrow_to_astropy(
            pyarrow.parquet.read_table(io.BytesIO(result["table_parquet"]))
        )
        print(f"table: {len(blitz)} rows x {len(blitz.colnames)} cols")
        print(blitz["det_name", "donut_id", "group_id", "group_fit_success"][:4])
        print("zk_deviation_ccs[0][:8]:", list(blitz["zk_deviation_ccs"][0][:8]))
    else:
        print("error:", resp.get("error"))

    parent_conn.send({"cmd": "shutdown"})
    print("shutdown ->", parent_conn.recv())
    proc.join(timeout=5)

    view.release()
    shm.close()
    shm.unlink()
