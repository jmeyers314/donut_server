"""The long-lived engine that runs the real wavefront pipeline.

Holds `_PREPARED_CACHE`, a small LRU of `PreparedEntry` bundles -- each one a
task, a calib set, and a resharded refcat, together under one composite key
(the `-c`/`-C` override list, band + calib selector, and the refcat's level-5
shard-id set, respectively). Bundled per entry, rather than the three
independent singleton globals this replaced, because a second `/prepare`
before any `/push` must not silently retarget a `job_id` that is still
waiting to be pushed: a `job_id`'s `prepared_key` names the exact entry that
was live when it was prepared, and `push` reactivates that entry -- reloading
it if it was since evicted -- rather than running whatever the *most recent*
prepare happened to load.

Per job, `run_job` rebuilds the raw exposures out of shared memory, hands them
to a hand-built in-memory Butler, and calls the real
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
# fork workers don't each fork with a live thread pool. bin.src/donutServer.py
# sets the same vars before importing anything that pulls numpy, which is what
# makes this order-independent in the real service; this block is the fallback
# for anything that imports coordinator directly.
# Verified at startup by _assert_single_threaded_blas.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import gc
import glob
import io
import logging
import multiprocessing as mp
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, NamedTuple

import pyarrow.parquet
import threadpoolctl

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
from lsst.pipe.base.configOverrides import ConfigOverrides
from lsst.pipe.base.tests.in_memory_limited_butler import InMemoryLimitedButler
from lsst.ts.wep.blitz.donutBlitzCorner import (
    DonutBlitzCornerConfig,
    DonutBlitzCornerTask,
)

_log = logging.getLogger(__name__)

NUM_WORKERS = 8

def calib_dir() -> str:
    """The calib directory, located by the caller's environment.

    Resolved per call rather than at import so that importing this module needs
    no data -- test_result_serialization drives the parquet path with a synthetic
    table and no calibs at all.
    """
    d = os.environ.get("DONUT_SERVER_CALIB_DIR")
    if not d:
        raise RuntimeError(
            "DONUT_SERVER_CALIB_DIR is not set; point it at the directory "
            "holding ptc_*.fits, linearizer_*.fits, flat_*.fits (see README)."
        )
    return d


def stamp_dir() -> str:
    """Where the image-bearing result tables are written.

    Required, and resolved per call, for the same two reasons as calib_dir():
    an unset data directory is a loud error rather than a silent empty result,
    and importing this module must need no data.

    Unlike the other three directories this one is an *output*, so the failure
    it guards against is different: the write happens after the push has already
    been answered, where a raise reaches nobody. _build_entry calls this so the
    error surfaces on /prepare instead.
    """
    d = os.environ.get("DONUT_SERVER_STAMP_DIR")
    if not d:
        raise RuntimeError(
            "DONUT_SERVER_STAMP_DIR is not set; point it at a writable directory "
            "for the per-job image tables (~17 MB each, unpruned -- see README)."
        )
    return d

# Per-donut postage stamps: ~21 MB of the result table (measured on the r-band
# exposure, 63 donuts -- stamp is 167x167 each, wf_img and model_img 83x83).
# Held out of the reply, not discarded: the remaining 59 columns, including all
# the Zernikes, are ~214 KB as parquet and are what the client waits for. The
# images are written to stamp_dir() after that reply -- see _DEFERRED.
IMAGE_COLUMNS = ("stamp", "wf_img", "model_img")

# Nothing here validates the instrument against a real Registry, and the blitz
# task never uses it functionally (its real visit id comes off the raw header).
INSTRUMENT = "LSSTCam"

# Arbitrary; no real collection is involved.
RUN = "donut_server"

# The reusable shared block the front-end streams raw pixels into, opened once
# in coordinator_main and read in place -- raw pixels never cross the Pipe.
_SHM: Any = None
_SHM_VIEW: Any = None

# One job's finished result table, handed from run_job to coordinator_main so the
# work that does not gate the reply can happen after it. run_job sets this; the
# loop drains it once the reply is on the wire and clears it unconditionally.
#
# This holds the *table*, not serialized bytes, because the deferred work is not
# only serialization: DonutBlitzPlotTask reads the same catalog (its plot subtask
# is always constructed, and savePlots defaults to False precisely so plots can
# be generated later from the in-memory results), and it needs the columns, not a
# parquet blob. One retained handle serves both.
#
# The retention is ~21 MB. The ~900 MB of exposures are still freed inside run_job
# before it returns, which is what the ~30 s cadence actually requires.
_DEFERRED: dict[str, Any] = {}

# Immortal: nothing can invalidate a DimensionUniverse(), so it is built once and
# shared by every PreparedEntry rather than carried inside one.
_UNIVERSE: Any = None

# How many distinct (task, calib, refcat) bundles to keep warm at once. Bounded
# rather than unbounded because a CalibSet is ~100 FITS files' worth of resident
# memory -- sized to "a couple of configs in flight", not measured against a real
# workload yet.
PREPARED_CACHE_CAP = 3


@dataclass
class PreparedEntry:
    """One fully-loaded (task, calib, refcat) bundle, keyed by all three of
    their own independent keys at once.

    Bundled together -- rather than as three independently-keyed globals --
    because a `job_id`'s `prepared_key` must name one thing that `push` can
    either find in the cache or rebuild whole; mixing today's cache-miss
    task with yesterday's cache-hit calib would be exactly the silent
    wrong-config run this cache exists to prevent.
    """

    key: tuple
    task: Any
    task_dump: str
    calib: Any  # a CalibSet, see below
    refcat_store: refcat_store.RefCatStore


# Insertion order is LRU order: touched entries are popped and re-inserted at
# the end (MRU), so the front is always the next eviction. dict/OrderedDict
# iteration order is why this needs no separate bookkeeping.
_PREPARED_CACHE: "OrderedDict[tuple, PreparedEntry]" = OrderedDict()

# The prepare command that built each still-known key, kept forever (never
# evicted alongside its PreparedEntry): a push-time cache miss must rebuild
# from the *exact* args that were live at prepare time, and a composite key's
# refcat component (a frozenset of shard ids) cannot be inverted back into the
# boresight that produced it. These dicts are tiny (a handful of floats and
# strings) next to a CalibSet, so keeping every key's command around costs
# nothing worth bounding.
_PREPARE_COMMANDS: dict[tuple, dict] = {}

# The composite key of whichever PreparedEntry is currently active -- i.e. the
# one _CALIB_STORE/_REFCAT_STORE/_TASK* below describe right now. None before
# the first successful prepare.
_ACTIVE_KEY: tuple | None = None

# Mirror the currently-active PreparedEntry's task/calib/refcat, so run_job and
# build_quantum_context need no change from the pre-cache design: they still
# just read _CALIB_STORE / _REFCAT_STORE / _TASK. Kept in sync by
# _activate_entry, the single place that switches which entry is "live".
_CALIB_STORE: dict[str, Any] = {}
_REFCAT_STORE = refcat_store.RefCatStore()
_TASK: Any = None
_TASK_DUMP: str = ""


class ConfigOverrideError(RuntimeError):
    """A supplied -c/-C override could not be applied.

    A distinct type so the front-end can answer 400 rather than 500 for operator
    error. Only `str(exc)` crosses the Pipe, so without this the alternative is
    sniffing AttributeError / FieldValidationError / SyntaxError text.
    """


class QuantumBundle(NamedTuple):
    """Everything runQuantum needs, plus the handles to read its output back."""

    butler_qc: Any
    input_refs: Any
    output_refs: Any
    butler: Any
    output_ref: Any
    timings: dict


def override_key(spec: list[dict] | None) -> tuple:
    """Hashable, order-sensitive identity of an override list.

    Order matters because `ConfigOverrides.applyTo` applies in insertion order and
    the last write wins, so the same entries in a different order are a different
    config. `name` is included because it is compiled into the code object and so
    changes tracebacks.

    Carries the override text in full rather than a hash: the front-end caps the
    total size, and a collision would silently serve a task built from a *different*
    config, which is the one failure here nobody could diagnose.

    None and [] both yield (), so "no overrides" has a single representation.
    """
    entries = []
    for entry in spec or ():
        if entry["kind"] == "value":
            entries.append(("value", entry["field"], entry["value"]))
        else:
            entries.append(("python", entry.get("name") or "<override>", entry["text"]))
    return tuple(entries)


def apply_overrides(config, spec: list[dict] | None) -> None:
    """Apply a -c/-C override list to `config`, in order, pipetask-style.

    This is `pipetask run`'s own mechanism: ConfigOverrides.addValueOverride is `-c`
    and addPythonOverride is `-C`, and a single applyTo() at the end preserves the
    relative order of the two kinds.

    A "value" entry's value stays a *string* all the way from the command line to
    here on purpose: applyTo only runs its expression parser on strings, and that
    parser is what gives `-c` its command-line semantics (bare words become strings,
    `[1,2]` becomes a list, `True` becomes a bool). Pre-parsing it to JSON types
    upstream would take the YAML branch instead and quietly change those semantics.

    A "python" entry is compiled with its client-side filename so a traceback names
    the operator's file rather than <string>. It is passed as a code object to
    addPythonOverride, which execs it -- deliberately never addFileOverride, which
    would resolve a path on *this* host.
    """
    overrides = ConfigOverrides()
    try:
        for index, entry in enumerate(spec or ()):
            if entry["kind"] == "value":
                overrides.addValueOverride(entry["field"], entry["value"])
            else:
                name = entry.get("name") or "<override>"
                overrides.addPythonOverride(compile(entry["text"], name, "exec"))
        overrides.applyTo(config)
    except Exception as exc:
        raise ConfigOverrideError(f"{type(exc).__name__}: {exc}") from exc


# What build_quantum_context knows how to wire. An override that perturbs the
# connection set past these is rejected at prepare rather than allowed to fail on
# the push path -- see _check_connections.
WIRED_INPUTS = frozenset(
    {"raws", "ptc", "linearizer", "crosstalk", "flat", "intrinsicZernikes", "refCat"}
)
WIRED_OUTPUTS = frozenset({"cornerResults"})


def _check_connections(config) -> None:
    """Refuse a config whose connection set this coordinator cannot wire.

    build_quantum_context builds exactly one output ref, for `cornerResults`, and a
    payload for exactly the inputs above. A config that asks for more is not a
    coordinator bug but it *looks* like one: `-c doZernikesOutput=True` adds a
    second output, and buildDatasetRefs then raises a bare `KeyError: 'zernikes'`
    from inside the task -- on the push path, after ~900 MB of raws have already
    crossed the wire.

    Checked here, at prepare, where the error can still reach a client and name the
    actual cause. Runs before the task ctor because building the connections is the
    cheaper half.
    """
    conns = config.connections.ConnectionsClass(config=config)
    inputs = set(conns.inputs) | set(conns.prerequisiteInputs)
    outputs = set(conns.outputs)

    if extra_out := sorted(outputs - WIRED_OUTPUTS):
        raise ConfigOverrideError(
            f"this service cannot wire output connection(s) {extra_out}: it builds a "
            f"ref only for {sorted(WIRED_OUTPUTS)}, so the task would fail on push "
            f"with a KeyError. Supporting these needs per-detector output refs in "
            f"build_quantum_context."
        )
    if extra_in := sorted(inputs - WIRED_INPUTS):
        raise ConfigOverrideError(
            f"this service cannot supply input connection(s) {extra_in}: no dataset "
            f"payload is built for them, so the task would fail on push."
        )


def _check_timeouts(config) -> None:
    """Refuse a hangTimeout that would kill this process mid-job.

    The hang watchdog fires from a side thread inside the coordinator and, having no
    way to know which unit is late, can only `os._exit(1)` the whole process. A
    hangTimeout below the per-unit timeout therefore converts every ordinary slow
    job into a coordinator death -- and the front-end's restart path makes that
    unrecoverable without operator action: a push-provoked loss is replayed by
    _reprime (its skip-guard only covers prepare-provoked losses), and a successful
    hello resets the restart counter, so it never reaches DEGRADED. The result is an
    indefinite one-kill-per-push loop that looks healthy between pushes.

    Non-positive is deliberately *allowed*: ts_wep treats `timeout <= 0` as
    "watchdog disabled", which is a legitimate thing to ask for. Only a positive
    value that undercuts unitTimeout is rejected -- the invariant the field's own
    docstring states ("Raise it alongside unitTimeout, never below it").
    """
    hang = config.hangTimeout
    unit = config.unitTimeout
    if hang is not None and hang > 0 and unit is not None and hang <= unit:
        raise ConfigOverrideError(
            f"hangTimeout={hang} must exceed unitTimeout={unit} (or be <= 0 to "
            "disable the watchdog): the watchdog can only abort the whole "
            "coordinator process, so a hangTimeout under the per-unit timeout "
            "turns every slow job into a coordinator restart."
        )


def build_task(spec: list[dict] | None) -> tuple[Any, str]:
    """Build a task from a fresh config plus `spec`. Returns (task, config dump).

    Touches no global, which is the point: applyTo mutates in place and stops at the
    first failing override, so a rejected override list must not be able to leave a
    half-mutated config installed. On any raise the caller's previous good task is
    still the live one.

    No explicit config.validate() -- Task.__init__ already calls it.
    """
    config = DonutBlitzCornerConfig()
    apply_overrides(config, spec)
    _check_connections(config)
    _check_timeouts(config)
    task = DonutBlitzCornerTask(config=config)
    return task, config.saveToString()


def require_task() -> tuple[Any, Any]:
    """(universe, task) for the push path. Never builds anything.

    Only prepare may change the task: task construction must happen before the
    task's own forks, and a push that silently built a default-config task would
    also be a push that silently ignored the operator's overrides.

    Normally unreachable -- run_job's band cross-check fires first and says more.
    Reachable in principle after a restart whose re-prime failed, which leaves the
    coordinator READY but unprimed while the front-end still holds PREPARED jobs
    that /push accepts. An explicit error rather than an AttributeError on None.
    """
    if _TASK is None:
        raise RuntimeError(
            "no task configured: /prepare has not run on this coordinator since it "
            "started (or its last prepare failed)"
        )
    return _UNIVERSE, _TASK


def task_config_dump() -> str:
    """The live task's full config as loadable Python. Served by GET /config."""
    return _TASK_DUMP


def _prepare_key(command: dict) -> tuple:
    """The composite cache key for one /prepare command.

    The refcat component is the *shard-id set*, not the raw boresight, so a
    dither within one pointing (same set) hits the cache -- matching
    RefCatStore's own reuse guard, which is keyed the same way.
    """
    task_key = override_key(command.get("config_overrides"))
    calib_key = (command["band"], command["calib_selector"])
    refcat_key = frozenset(
        refcat_store.shard_ids_for_pointing(command["boresight_ra"], command["boresight_dec"])
    )
    return (task_key, calib_key, refcat_key)


def _activate_entry(entry: "PreparedEntry") -> None:
    """Make `entry` the one run_job/build_quantum_context see.

    The single place that touches the mirror globals, so every path that
    switches the live config -- a fresh prepare, a cache-hit prepare, or a
    push-time reload -- goes through here and cannot leave them half-updated.
    """
    global _ACTIVE_KEY, _TASK, _TASK_DUMP, _REFCAT_STORE
    _ACTIVE_KEY = entry.key
    _TASK = entry.task
    _TASK_DUMP = entry.task_dump
    _CALIB_STORE.clear()
    _CALIB_STORE["config_key"] = entry.calib.config_key
    _CALIB_STORE["calib"] = entry.calib
    _REFCAT_STORE = entry.refcat_store


def _evict_lru() -> None:
    """Drop cache entries beyond PREPARED_CACHE_CAP, oldest (least-recently
    touched) first. `_PREPARE_COMMANDS` is deliberately not pruned here -- see
    its own docstring."""
    while len(_PREPARED_CACHE) > PREPARED_CACHE_CAP:
        evicted_key, _ = _PREPARED_CACHE.popitem(last=False)
        _log.info(
            "prepared-cache: evicted %r (cap=%d, now holding %d)",
            evicted_key, PREPARED_CACHE_CAP, len(_PREPARED_CACHE),
        )


def _build_entry(key: tuple, command: dict) -> tuple["PreparedEntry", dict]:
    """Build a fresh PreparedEntry for `key` from `command`'s args.

    Used both for a cold /prepare and for a push-time reload of an evicted
    entry -- the two are the same operation, just triggered differently.
    """
    spec = command.get("config_overrides")

    t0 = time.monotonic()
    task, task_dump = build_task(spec)
    task_timing = {
        "reused": False,
        "elapsed_s": time.monotonic() - t0,
        "n_overrides": len(key[0]),
    }

    # Pay the pyarrow init here, not on push.
    _warm_up_parquet()
    # The only place an unset DONUT_SERVER_STAMP_DIR can be reported to a client:
    # the write itself happens after the push reply, where a raise reaches nobody.
    os.makedirs(stamp_dir(), exist_ok=True)
    calib, calib_timing = _build_calib(command["band"], command["calib_selector"])

    store = refcat_store.RefCatStore()
    refcat_timing = store.ensure(command["boresight_ra"], command["boresight_dec"])

    entry = PreparedEntry(key=key, task=task, task_dump=task_dump, calib=calib, refcat_store=store)
    return entry, {"task": task_timing, "calib": calib_timing, "refcat": refcat_timing}


def ensure_prepared(command: dict) -> dict:
    """Get-or-build the PreparedEntry for one /prepare command, activate it,
    and return its key plus prepare timings shaped like today's response.

    On a cache hit, `RefCatStore.ensure` is still called (against the possibly
    slightly-different boresight this call carries) so the response always
    reports the current pointing and stays cheap on the reused path -- same
    behaviour as the old singleton `_REFCAT_STORE.ensure` reuse guard.
    """
    global _UNIVERSE
    if _UNIVERSE is None:
        _UNIVERSE = DimensionUniverse()

    key = _prepare_key(command)
    entry = _PREPARED_CACHE.get(key)

    if entry is None:
        entry, timings = _build_entry(key, command)
        _PREPARED_CACHE[key] = entry
        _PREPARE_COMMANDS[key] = dict(command)
        _evict_lru()
    else:
        _PREPARED_CACHE.move_to_end(key)
        timings = {
            "task": {"reused": True, "elapsed_s": 0.0, "n_overrides": len(key[0])},
            "calib": {"reused": True, "elapsed_s": 0.0},
            "refcat": entry.refcat_store.ensure(command["boresight_ra"], command["boresight_dec"]),
        }

    _activate_entry(entry)
    return {"key": key, "timings": timings}


def ensure_prepared_for_push(key: tuple) -> "PreparedEntry":
    """The PreparedEntry for `key`, for the push path. Reloads it from its
    original prepare command if it was evicted since, rather than silently
    running whatever entry happens to be active -- the exact hazard this cache
    replaced. Moves the entry to MRU either way, so an active job's config is
    not evicted out from under a straggling push.
    """
    entry = _PREPARED_CACHE.get(key)
    if entry is not None:
        _PREPARED_CACHE.move_to_end(key)
        _activate_entry(entry)
        return entry

    command = _PREPARE_COMMANDS.get(key)
    if command is None:
        raise RuntimeError(
            f"no prepared config found for key {key!r}: it was never prepared on "
            "this coordinator (or the coordinator has restarted since)"
        )
    _log.info("prepared-cache: reloading evicted entry for push, key=%r", key)
    entry, _ = _build_entry(key, command)
    _PREPARED_CACHE[key] = entry
    _evict_lru()
    _activate_entry(entry)
    return entry


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


def write_stamp_table(job_id: str, table) -> str:
    """Write the image-bearing table to stamp_dir(), atomically. Returns the path.

    Via a temp file plus os.replace, so the visible path only ever names a
    complete file. The front-end serves it with no completion signal from this
    process, and nothing here can tell it a write was cut short: /admin/restart
    SIGKILLs this process, and by then the job is already DONE as far as the
    front-end and the client are concerned. Hence the rule that existence means
    completeness. An interrupted write leaves an inert .tmp instead.
    """
    # No makedirs here: _build_entry already created the directory, and a push
    # cannot reach this without a prepare having succeeded first.
    final = os.path.join(stamp_dir(), f"{job_id}.parquet")
    tmp = f"{final}.tmp"
    payload = to_parquet(table)
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    return final


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


def _build_calib(band: str, calib_selector: str) -> tuple[CalibSet, dict]:
    """Load one CalibSet fresh from disk. Always builds -- the composite-key
    cache above this is where reuse is decided, so this need not guard itself."""
    config_key = (band, calib_selector)
    t0 = time.monotonic()

    calibs = calib_dir()
    detector_ids = _discover_detector_ids(calibs)

    ptc_by_name: dict = {}
    linearizer_by_name: dict = {}
    crosstalk_by_name: dict = {}
    flat_by_name: dict = {}
    intrinsic_zernikes_by_name: dict = {}

    for det_id in detector_ids:
        # Band-independent, required. Detector name comes off the calib
        # object itself (._detectorName) -- no separate id->name table needed.
        ptc = ipIsr.PhotonTransferCurveDataset.readFits(
            os.path.join(calibs, f"ptc_{det_id}.fits")
        )
        name = ptc._detectorName
        ptc_by_name[name] = ptc
        linearizer_by_name[name] = ipIsr.Linearizer.readFits(
            os.path.join(calibs, f"linearizer_{det_id}.fits")
        )
        crosstalk_by_name[name] = ipIsr.CrosstalkCalib.readFits(
            os.path.join(calibs, f"crosstalk_{det_id}.fits")
        )

        # Band-dependent, required.
        flat_path = os.path.join(calibs, f"flat_{det_id}_{band}.fits")
        if not os.path.exists(flat_path):
            raise RuntimeError(
                f"Missing required flat calib for detector {name} ({det_id}), "
                f"band {band!r}: {flat_path}"
            )
        flat_by_name[name] = afwImage.ExposureF.readFits(flat_path)

        # Band-dependent, optional (mirrors the real Butler connection's
        # minimum=0): missing file just means no entry for this detector.
        iz_path = os.path.join(calibs, f"intrinsicZernikes_{det_id}_{band}.fits")
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

    timings = {
        "reused": False,
        "elapsed_s": time.monotonic() - t0,
        "n_detectors": len(detector_ids),
        "n_intrinsic_zernikes": len(intrinsic_zernikes_by_name),
        "band": band,
    }
    return calib, timings


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


def build_quantum_context(
    exposures: dict[str, Any], calib: CalibSet, num_workers: int | None = None
) -> QuantumBundle:
    """Build the InMemoryLimitedButler / Quantum / in+out refs for one exposure.

    No Registry, SQLite or obs_lsst camera package is involved: the dimension
    universe, dataset types, data coordinates and quantum are all hand-built.
    The result is ready to hand straight to
    `DonutBlitzCornerTask.runQuantum()`.
    """
    universe, task = require_task()
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
        butler, quantum, resources=ExecutionResources(num_cores=num_workers or NUM_WORKERS)
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

    # Rows the task never grouped carry group_id "", which is not a group: it
    # would otherwise show up as one permanently-failed group and drag the
    # succeeded/total ratio to (n-1)/n on every job.
    groups: dict[Any, bool] = {}
    for group_id, success in zip(table["group_id"], table["group_fit_success"]):
        group_id = str(group_id)
        if not group_id.strip():
            continue
        groups[group_id] = bool(success)

    return {
        "n_rows": len(table),
        "n_detectors": len(per_detector),
        "rows_per_detector": dict(sorted(per_detector.items())),
        "n_groups": len(groups),
        "n_groups_succeeded": sum(groups.values()),
        "columns": list(table.colnames),
    }


def run_job(job_id: str, layout: list, num_workers: int | None = None) -> dict:
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

    bundle = build_quantum_context(exposures, calib, num_workers)
    _, task = require_task()

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
    summary["deferred_columns"] = list(IMAGE_COLUMNS)
    summary["parquet_bytes"] = len(parquet_bytes)

    bundle_timings = bundle.timings
    quantum_info = bundle_timings.pop("quantum")

    # Hand the full table to coordinator_main, which picks it up *after* replying.
    # Anything done to it here would be on the critical path the client is blocked
    # on; everything it is needed for can wait ~20 s for the next exposure.
    _DEFERRED["job_id"] = job_id
    _DEFERRED["table"] = full_table

    # Reclaim this job's cycles now rather than whenever the automatic collector
    # next fires: the ~900 MB of exposures should be gone well before the next
    # exposure arrives (~30 s cadence). full_table is deliberately not among them
    # -- dropping the local name leaves _DEFERRED's ~21 MB reference, which is the
    # point -- but the exposures and the butler holding them still go now.
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

    Uses threadpoolctl rather than ctypes.CDLL(None) to read the thread counts:
    on Linux the loader opens libopenblas/libomp RTLD_LOCAL, so their symbols
    aren't visible through a global handle even though the libraries are
    loaded (confirmed: openblas_get_num_threads is exported by
    libopenblasp*.so but unreachable via CDLL(None), raising
    `undefined symbol`). macOS's flat namespace masked this. threadpoolctl
    resolves each library's own path and dlopen()s it directly, so it works on
    both platforms.
    """
    hot = {
        info["prefix"]: info["num_threads"]
        for info in threadpoolctl.threadpool_info()
        if info["num_threads"] != 1
    }
    if hot:
        raise RuntimeError(
            f"thread pools not clamped: {hot}. Something imported numpy before "
            "coordinator, so the thread-limit env vars never bound. Export them "
            "before starting the process (see bin.src/donutServer.py)."
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


def _run_deferred() -> None:
    """Do the finished job's non-urgent work, then drop the table.

    Called by coordinator_main *after* the push reply is on the wire, so none of
    this is on the latency path the client sees. It does delay the next command,
    since the loop is serial and does not return to recv() until this finishes --
    at ~0.1 s against a ~30 s cadence that is noise, and there is deliberately no
    command timeout on the front-end to misread it as a hang.

    Every failure is logged and swallowed. Raising is not an option that leads
    anywhere useful: the client already holds a 200 for this job, and letting the
    exception escape would kill the coordinator over stamps that nobody is
    blocked on. An unset output directory is caught at prepare time instead,
    where it can still reach a client.
    """
    table = _DEFERRED.pop("table", None)
    job_id = _DEFERRED.pop("job_id", None)
    if table is None:
        return
    try:
        t0 = time.perf_counter()
        path = write_stamp_table(job_id, table)
        _log.info(
            "job %s: wrote %s (%.1f MB) in %.3fs",
            job_id, path, os.path.getsize(path) / 1e6, time.perf_counter() - t0,
        )
    except Exception:
        # exc_info, because this is the only record that will ever exist of it.
        _log.exception("job %s: deferred image write failed", job_id)
    finally:
        # Explicit, so the ~21 MB goes now rather than on the next job's dict
        # assignment -- which would otherwise hold two tables at once.
        del table
        _DEFERRED.clear()


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
                    prepared = ensure_prepared(command)
                    # config_dump sits beside `timings`, never inside it: the
                    # front-end stores timings on every JobRecord and returns them
                    # 50 rows at a time from /admin/jobs at 1 Hz, where ~77 KB a row
                    # would be catastrophic. Sent on the reused path too, so a
                    # re-prime that hit the guard still refreshes the cache.
                    #
                    # The front-end never inspects prepared_key -- it is opaque
                    # data to store on the JobRecord and echo back verbatim on
                    # push, since this Pipe is pickle (unlike the front-end's own
                    # HTTP boundary) it survives the round trip unchanged.
                    conn.send({
                        "ok": True,
                        "timings": prepared["timings"],
                        "config_dump": task_config_dump(),
                        "prepared_key": prepared["key"],
                    })
                except Exception as exc:
                    # Tagged so the front-end can answer 400 for operator error and
                    # 500 for everything else; only str(exc) crosses this Pipe.
                    conn.send({
                        "ok": False,
                        "error": str(exc),
                        "kind": "config_override" if isinstance(exc, ConfigOverrideError) else None,
                    })
            elif cmd == "push":
                try:
                    prepared_key = command.get("prepared_key")
                    if prepared_key is not None:
                        # Reactivates the job's own entry -- reloading it first if
                        # it was evicted since prepare -- rather than running
                        # whatever entry a later prepare happened to leave active.
                        ensure_prepared_for_push(tuple(prepared_key))
                    result = run_job(
                        command["job_id"], command["layout"], command.get("num_workers")
                    )
                    conn.send({"ok": True, "result": result})
                except Exception as exc:
                    conn.send({"ok": False, "error": str(exc)})
                # After the reply, deliberately: the client is unblocked by the
                # send above, so everything here is spent against the ~20 s gap
                # before the next exposure rather than against the push. Also
                # runs on the failure path, where _DEFERRED is simply empty.
                _run_deferred()
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
    source = client.resolve_from_files(client.raw_dir(), VISIT)
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
        "config_overrides": [],
    })
    prepare_resp = parent_conn.recv()
    # The config dump is ~77 KB, so report its size rather than printing it.
    dump = prepare_resp.pop("config_dump", "")
    print("prepare ->", prepare_resp, f"config_dump={len(dump)} bytes")

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

    # Deliberately checked after the shutdown round-trip, not after the push
    # reply: the image write happens between the push reply and the next recv, so
    # right after the reply is exactly when it is still in flight. The shutdown
    # reply is the first thing that cannot arrive until the write has finished --
    # which is also the property the whole design rests on, so seeing the file
    # missing above and present below is the point, not a quirk of the test.
    parent_conn.send({"cmd": "shutdown"})
    print("shutdown ->", parent_conn.recv())
    proc.join(timeout=5)

    stamps = os.path.join(stamp_dir(), "smoke-1.parquet")
    if os.path.exists(stamps):
        print(f"images: {stamps} ({os.path.getsize(stamps) / 1e6:.1f} MB)")
    else:
        print(f"images: MISSING at {stamps} -- check the log for the write error")

    view.release()
    shm.close()
    shm.unlink()
