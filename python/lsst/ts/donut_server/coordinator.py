"""The long-lived engine that runs the real wavefront pipeline.

Holds `_PREPARED_CACHE`, a small LRU of `PreparedEntry` bundles -- each one a
task plus a resharded refcat, under one composite key (the `-c`/`-C` override
list, the physical_filter, and the refcat's level-5 shard-id set). Bundled per
entry, rather than the independent singleton globals this replaced, because a second
`/prepare` before any `/push` must not silently retarget a `job_id` that is
still waiting to be pushed: a `job_id`'s `prepared_key` names the exact entry
that was live when it was prepared, and `push` reactivates that entry --
reloading it if it was since evicted -- rather than running whatever the *most
recent* prepare happened to load.

Calibs are cached separately, and in two halves: `_FILTER_CALIB_CACHE` for the
flats and intrinsic Zernikes, and `_DETECTOR_CALIB_CACHE` -- one slot -- for the
PTCs, linearizers and crosstalk, which depend on nothing in `/prepare` at all.
They are by far the most expensive thing prepare loads, so a pointing change
must not discard them and a filter change must discard only the half that the
filter selects. `_activate_entry` composes the two into the flat `CalibSet` view
its readers expect. The anti-hazard argument above is unaffected -- it never
required that calibs be *copied* per entry, only that a `job_id` resolve to the
exact config it was prepared against, which a lookup off its own prepare command
gives exactly.

Where the calibs come from is a backend: `FilesCalibBackend` reads flat FITS out
of `DONUT_SERVER_CALIB_DIR` (the default, and what the tests use -- it needs no
repo), and `ButlerCalibBackend` resolves them from a real repo the way
`pipetask run` would. The backend answers an *identity* per half, and that
identity is what the two caches are keyed on, so a recertification that changes
which datasets a filter resolves to invalidates the right half by itself.

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
    Butler,
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionUniverse,
    Quantum,
    Timespan,
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

def _num_workers() -> int:
    """How many cores the task may fork over, per push.

    Deliberately *not* divided by the number of flights: 8 workers on this host's
    128 cores means even N=4 is nowhere near oversubscription, so partitioning the
    budget would halve per-job parallelism to avoid a problem two orders of
    magnitude away. Configurable so a small host can lower it. Resolved at import
    in the child, which inherits the front-end's environment.
    """
    try:
        n = int(os.environ.get("DONUT_SERVER_NUM_WORKERS", "8"))
    except ValueError:
        return 8
    return max(1, n)


NUM_WORKERS = _num_workers()


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
            "holding ptc_*.fits, linearizer_*.fits, flat_*.fits (see README). "
            "Required only under DONUT_SERVER_CALIB_BACKEND=files (the default)."
        )
    return d


# The eight corner wavefront sensors, id -> full_name. SW0 is extra-focal, SW1
# intra-focal. Verified against the `detector` dimension records in
# /Users/jmeyers3/repo; scripts/export_blitz_repo.py has the same id list under
# the name CORNER_DETECTORS, but scripts/ is not on the package path so this is
# a deliberate second copy rather than an import.
#
# The files backend does not use this -- it discovers ids from the ptc_*.fits
# filenames, which is authoritative for a tree that might hold a subset. The
# butler backend has no files to glob, so the focal plane has to be named.
CORNER_DETECTOR_NAMES = {
    191: "R00_SW0",
    192: "R00_SW1",
    195: "R04_SW0",
    196: "R04_SW1",
    199: "R40_SW0",
    200: "R40_SW1",
    203: "R44_SW0",
    204: "R44_SW1",
}
CORNER_DETECTORS = tuple(CORNER_DETECTOR_NAMES)


def butler_repo() -> str:
    """The butler repo for the `butler` calib backend, from the environment."""
    d = os.environ.get("DONUT_SERVER_BUTLER_REPO")
    if not d:
        raise RuntimeError(
            "DONUT_SERVER_BUTLER_REPO is not set, but "
            "DONUT_SERVER_CALIB_BACKEND=butler requires it; point it at a "
            "butler repo root or an alias (see README)."
        )
    return d


def butler_collections() -> list[str]:
    """The input collections to resolve calibs from, highest priority first.

    Comma-separated, defaulting to the same `LSSTCam/defaults` that
    client.py's --collections and scripts/export_blitz_repo.py use. Passed to
    the butler *unfiltered* -- see ButlerCalibBackend on why the chain is not
    reduced to its CALIBRATION children.
    """
    raw = os.environ.get("DONUT_SERVER_BUTLER_COLLECTIONS") or "LSSTCam/defaults"
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise RuntimeError(
            f"DONUT_SERVER_BUTLER_COLLECTIONS is set to {raw!r}, which names no "
            "collections; unset it for the default (LSSTCam/defaults)."
        )
    return names


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

# Per-donut postage stamps: ~21 MB of the result table (measured on the r_57
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

# How many distinct (task, refcat) bundles to keep warm at once. Bounded rather
# than unbounded because a resharded refcat set is a few hundred MB -- sized to
# "a couple of configs in flight", not measured against a real workload yet.
PREPARED_CACHE_CAP = 3

# How many filters' calibs to keep warm. A filter change is rare, so two keeps
# the current and previous one resident. This bound is what caps the
# filter-dependent calib RSS: entries in _PREPARED_CACHE deliberately hold no
# reference to a CalibSet, so _FILTER_CALIB_CACHE is its sole owner and dropping
# one here really frees it.
CALIB_CACHE_CAP = 2


class PrepareKey(NamedTuple):
    """The composite cache key for one /prepare command.

    A NamedTuple rather than a bare tuple so the components are read by name:
    `_activate_entry` needs the filter out of the middle of the key, and a
    positional `key[1]` there is the kind of thing nothing type-checks and a
    later reordering silently breaks. Still a tuple, so it hashes and compares
    exactly as the plain tuple it replaced -- `_PREPARED_CACHE` and
    `_PREPARE_COMMANDS` need no change.
    """

    task: tuple
    physical_filter: str
    refcat: frozenset
    # Part of the key because it selects calibs: under the butler backend two
    # prepares at different times can resolve to different datasets, and sharing
    # one entry between them would be the silent wrong-config run this cache
    # exists to prevent. Under files it never changes the calibs, so it only
    # costs key-space -- which is the right way round for a correctness guard.
    calib_time: str


@dataclass
class PreparedEntry:
    """One fully-loaded (task, refcat) bundle, keyed by the composite key.

    Bundled together -- rather than as independently-keyed globals -- because a
    `job_id`'s `prepared_key` must name one thing that `push` can either find in
    the cache or rebuild whole; mixing today's cache-miss task with yesterday's
    cache-hit refcat would be exactly the silent wrong-config run this cache
    exists to prevent.

    Carries no calib: the filter is already `key.physical_filter`, so the calib
    is resolved through the two calib caches at activation time. That keeps an
    entry cheap and keeps `CALIB_CACHE_CAP` a real memory bound rather than a
    lower bound.
    """

    key: PrepareKey
    task: Any
    task_dump: str
    refcat_store: refcat_store.RefCatStore


# Insertion order is LRU order: touched entries are popped and re-inserted at
# the end (MRU), so the front is always the next eviction. dict/OrderedDict
# iteration order is why this needs no separate bookkeeping.
_PREPARED_CACHE: "OrderedDict[PrepareKey, PreparedEntry]" = OrderedDict()

# Calibs are cached apart from _PREPARED_CACHE because they depend only on the
# filter: sharing one composite key would make a slew that shifts the refcat
# shard set reload every FITS file in calib_dir(). They are then cached in two
# pieces, because "only the filter" overstates it for three of the five types --
# the PTCs, linearizers and crosstalk depend on nothing in /prepare at all, and
# folding them in made a filter change re-read bytes that had not changed --
# most of the rebuild time for a small minority of the bytes, since they are
# deserialization-bound rather than I/O-bound.
#
# So the filter-independent half gets a single slot: it is the small half by
# memory and is identical across filters, so a second slot would buy nothing a
# first does not. `(identity, DetectorCalibs)` -- the identity is the backend's,
# and the slot is replaced when it changes, which under the files backend is
# never (a fixed calib_dir() cannot vary) and under the butler backend means a
# recertification. None until the first prepare builds it.
_DETECTOR_CALIB_CACHE: "tuple[Any, DetectorCalibs] | None" = None

# The filter-dependent half, on the same insertion-order-is-LRU idiom as
# _PREPARED_CACHE. Keyed on the backend's identity for this half, which under
# the files backend *is* the physical_filter string. This is the bulk of the
# calib memory, and so what CALIB_CACHE_CAP actually bounds.
_FILTER_CALIB_CACHE: "OrderedDict[Any, FilterCalibs]" = OrderedDict()

# The prepare command that built each still-known key, kept forever (never
# evicted alongside its PreparedEntry): a push-time cache miss must rebuild
# from the *exact* args that were live at prepare time, and a composite key's
# refcat component (a frozenset of shard ids) cannot be inverted back into the
# boresight that produced it. These dicts are tiny (a handful of floats and
# strings) next to a CalibSet, so keeping every key's command around costs
# nothing worth bounding.
_PREPARE_COMMANDS: dict[PrepareKey, dict] = {}

# The composite key of whichever PreparedEntry is currently active -- i.e. the
# one _CALIB_STORE/_REFCAT_STORE/_TASK* below describe right now. None before
# the first successful prepare.
_ACTIVE_KEY: PrepareKey | None = None

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


class CalibTimeError(RuntimeError):
    """The supplied `calib_time` could not be parsed.

    A distinct type for the same reason as ConfigOverrideError: operator error
    deserves a 400, and the front-end sees only `str(exc)`. The front-end cannot
    validate this itself -- parsing needs astropy, which it deliberately does
    not import -- so shape validation there and parse validation here is the
    split, and this is how the second half reports.
    """


class CalibConfigError(RuntimeError):
    """The configured calib backend cannot work, whatever the request.

    Raised only from `check_calib_config`, before the hello, and distinct from
    the two above because it is not about one request: a mistyped repo path or a
    collection chain that names nothing is wrong for every /prepare that will
    ever arrive, so there is no child a restart could produce that would behave
    differently. The front-end answers it by refusing to retry and taking the
    server down -- see Coord._bring_up.
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

    Normally unreachable -- run_job's filter cross-check fires first and says more.
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


def _prepare_key(command: dict) -> PrepareKey:
    """The composite cache key for one /prepare command.

    The refcat component is the *shard-id set*, not the raw boresight, so a
    dither within one pointing (same set) hits the cache -- matching
    RefCatStore's own reuse guard, which is keyed the same way.

    `physical_filter` and `calib_time` are subscripted, not `.get`: a caller that
    still sends the old `band` field, or no time at all, fails here loudly rather
    than keying the cache on None and only failing later on a
    `flat_<det>_None.fits` that does not exist, or on a butler lookup that
    quietly excludes every calibration collection.
    """
    task_key = override_key(command.get("config_overrides"))
    refcat_key = frozenset(
        refcat_store.shard_ids_for_pointing(command["boresight_ra"], command["boresight_dec"])
    )
    return PrepareKey(
        task_key, command["physical_filter"], refcat_key, command["calib_time"]
    )


def _ensure_detector_calibs(
    backend, ids: "CalibIds"
) -> tuple["DetectorCalibs", bool, float]:
    """Get-or-build the filter-independent calibs for `ids`, with whether they
    were reused and the seconds spent building.

    One slot, keyed on the backend's identity for this half -- i.e. an LRU of
    size 1. Under files that identity is a constant, so this loads exactly once
    per process as it always has; under the butler a recertification changes the
    UUID set and replaces the slot. The slot is the small half by memory and the
    one that is identical across filters, so a second slot would buy nothing a
    first does not.

    Assignment happens only after the load returns, so a failed load leaves the
    previous entry cached and the next prepare retries rather than serving a
    half-built set."""
    global _DETECTOR_CALIB_CACHE
    if _DETECTOR_CALIB_CACHE is not None and _DETECTOR_CALIB_CACHE[0] == ids.detector:
        return _DETECTOR_CALIB_CACHE[1], True, 0.0

    t0 = time.monotonic()
    detector = backend.load_detector(ids)
    elapsed_s = time.monotonic() - t0
    if _DETECTOR_CALIB_CACHE is not None:
        _log.info("calib-cache: detector-half identity changed, replacing the slot")
    _DETECTOR_CALIB_CACHE = (ids.detector, detector)
    return detector, False, elapsed_s


def _ensure_filter_calibs(
    backend, ids: "CalibIds", detector: "DetectorCalibs"
) -> tuple["FilterCalibs", bool, float]:
    """Get-or-build the filter-dependent calibs for `ids`, with whether they
    were reused and the seconds spent building.

    Keyed on the backend's identity for this half, which under files *is* the
    physical_filter string and under the butler is the resolved UUID set.

    `detector` is required rather than resolved here so the two halves cannot be
    loaded in the wrong order: the flats are keyed by detector name, which on
    the files path only the ptc headers know.

    Evicts past CALIB_CACHE_CAP, oldest first. Insertion happens only after the
    load returns, so a failed load leaves the previous filters cached."""
    filtered = _FILTER_CALIB_CACHE.get(ids.filtered)
    if filtered is not None:
        _FILTER_CALIB_CACHE.move_to_end(ids.filtered)
        return filtered, True, 0.0

    t0 = time.monotonic()
    filtered = backend.load_filter(ids, detector)
    elapsed_s = time.monotonic() - t0
    _FILTER_CALIB_CACHE[ids.filtered] = filtered
    while len(_FILTER_CALIB_CACHE) > CALIB_CACHE_CAP:
        evicted, _ = _FILTER_CALIB_CACHE.popitem(last=False)
        _log.info(
            "calib-cache: evicted filter-half %r (cap=%d, now holding %d)",
            evicted, CALIB_CACHE_CAP, len(_FILTER_CALIB_CACHE),
        )
    return filtered, False, elapsed_s


def _ensure_calib(command: dict) -> tuple["CalibSet", dict]:
    """Get-or-build both calib halves for one /prepare command and compose the
    CalibSet view over them, reporting the pair in a shared timing shape so a
    reused calib is described with its real counts rather than zeroed
    placeholders.

    Resolution runs on every call, before either cache is consulted, because it
    is what *produces* the cache keys: asking the backend which datasets this
    command means is the only way to know whether what is cached still answers
    it. It is deliberately cheap and reads no pixels, so a hit pays the question
    and not the answer.

    `elapsed_s` sums only what was actually built, and `reused` means *nothing*
    was: the per-half flags are what distinguish "a filter change reloaded the
    flats" from "a cold start reloaded everything".
    """
    backend = calib_backend()

    t0 = time.monotonic()
    ids = backend.resolve(command)
    resolve_s = time.monotonic() - t0

    detector, detector_reused, detector_s = _ensure_detector_calibs(backend, ids)
    filtered, filter_reused, filter_s = _ensure_filter_calibs(backend, ids, detector)
    calib = CalibSet(detector=detector, filtered=filtered)
    return calib, {
        "reused": detector_reused and filter_reused,
        "detector_reused": detector_reused,
        "filter_reused": filter_reused,
        "elapsed_s": detector_s + filter_s,
        "resolve_s": resolve_s,
        "backend": backend.name,
        "n_detectors": len(calib.detector_ids),
        "n_intrinsic_zernikes": len(calib.intrinsic_zernikes_by_name),
        "physical_filter": calib.physical_filter,
        "calib_time": command["calib_time"],
    }


def _activate_entry(entry: "PreparedEntry", command: dict) -> dict:
    """Make `entry` the one run_job/build_quantum_context see, and return the
    calib timings for it.

    The single place that touches the mirror globals, so every path that
    switches the live config -- a fresh prepare, a cache-hit prepare, or a
    push-time reload -- goes through here and cannot leave them half-updated.
    It is also the only caller of `_ensure_calib`, so one prepare resolves the
    calibs exactly once: resolution is a round trip to the registry under the
    butler backend, and doing it both here and in `ensure_prepared` would double
    every prepare's registry traffic to reach the same answer.

    `command` is the prepare command this activation is for -- the live one on a
    prepare, or the entry's original on a push-time reload. Which it is does not
    change the calibs: the key pins `physical_filter` and `calib_time`, and those
    are the only two fields the calibs depend on.
    """
    global _ACTIVE_KEY, _TASK, _TASK_DUMP, _REFCAT_STORE
    calib, calib_timing = _ensure_calib(command)
    _ACTIVE_KEY = entry.key
    _TASK = entry.task
    _TASK_DUMP = entry.task_dump
    _CALIB_STORE.clear()
    _CALIB_STORE["physical_filter"] = calib.physical_filter
    _CALIB_STORE["calib"] = calib
    _REFCAT_STORE = entry.refcat_store
    return calib_timing


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


def _build_entry(key: "PrepareKey", command: dict) -> tuple["PreparedEntry", dict]:
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
        "n_overrides": len(key.task),
    }

    # Pay the pyarrow init here, not on push.
    _warm_up_parquet()
    # The only place an unset DONUT_SERVER_STAMP_DIR can be reported to a client:
    # the write itself happens after the push reply, where a raise reaches nobody.
    os.makedirs(stamp_dir(), exist_ok=True)

    store = refcat_store.RefCatStore()
    refcat_timing = store.ensure(command["boresight_ra"], command["boresight_dec"])

    entry = PreparedEntry(key=key, task=task, task_dump=task_dump, refcat_store=store)
    return entry, {"task": task_timing, "refcat": refcat_timing}


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
            "task": {"reused": True, "elapsed_s": 0.0, "n_overrides": len(key.task)},
            "refcat": entry.refcat_store.ensure(command["boresight_ra"], command["boresight_dec"]),
        }

    timings["calib"] = _activate_entry(entry, command)
    return {"key": key, "timings": timings}


def ensure_prepared_for_push(key) -> "PreparedEntry":
    """The PreparedEntry for `key`, for the push path. Reloads it from its
    original prepare command if it was evicted since, rather than silently
    running whatever entry happens to be active -- the exact hazard this cache
    replaced. Moves the entry to MRU either way, so an active job's config is
    not evicted out from under a straggling push.

    `key` may arrive as a plain tuple -- the dispatcher rebuilds it from what
    crossed the Pipe -- so it is coerced back to a PrepareKey. It would hash and
    compare equal either way, but a rebuild on the evicted path reads the key by
    name, and a bare tuple would only fail there: the rarest path, and the one
    whose whole job is not to run under the wrong config.
    """
    key = PrepareKey(*key)
    # Looked up before the cache, and required either way: activation resolves
    # the calibs against the command this entry was prepared with, which is the
    # whole point on this path -- a push must not pick up a later prepare's
    # calibs any more than it picks up its task.
    command = _PREPARE_COMMANDS.get(key)
    if command is None:
        raise RuntimeError(
            f"no prepared config found for key {key!r}: it was never prepared on "
            "this coordinator (or the coordinator has restarted since)"
        )

    entry = _PREPARED_CACHE.get(key)
    if entry is not None:
        _PREPARED_CACHE.move_to_end(key)
        _activate_entry(entry, command)
        return entry

    _log.info("prepared-cache: reloading evicted entry for push, key=%r", key)
    entry, _ = _build_entry(key, command)
    _PREPARED_CACHE[key] = entry
    _evict_lru()
    _activate_entry(entry, command)
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
class DetectorCalibs:
    """The calibs dimensioned by (instrument, detector) alone, keyed by name.

    Nothing in /prepare selects these, so they are loaded once and kept -- see
    `_DETECTOR_CALIB_CACHE`. `build_quantum_context` puts exactly these three on
    `detector_did`, which is where the split line comes from.

    `names_by_id` is the id->name mapping the ptc headers established while
    loading. The filter-dependent half needs it -- flats are named by id on disk
    but keyed by name in memory -- and this is the only place it is known.
    """

    detector_ids: list
    names_by_id: dict
    ptc_by_name: dict
    linearizer_by_name: dict
    crosstalk_by_name: dict


@dataclass
class FilterCalibs:
    """The calibs that also carry physical_filter, keyed by detector name.

    physical_filter rather than band because that is what actually selected the
    flats -- band is a 6-valued coarsening of it, so two filters sharing a band
    (an `r_57` -> `r_03` swap) would be indistinguishable. `build_quantum_context`
    reads both labels off the exposure's FilterLabel, so there is no band field
    here to keep consistent with anything.

    intrinsic_zernikes_by_name is partial: a detector missing that file simply
    has no entry (mirrors the real Butler connection's minimum=0).
    """

    physical_filter: str
    flat_by_name: dict
    intrinsic_zernikes_by_name: dict


class CalibSet(NamedTuple):
    """The two calib halves presented as the one flat object readers expect.

    A view, not a container: it owns nothing and copies nothing, so composing a
    fresh one per activation is free and `CALIB_CACHE_CAP` keeps bounding real
    calib RSS through `_FILTER_CALIB_CACHE` alone. The forwarding properties
    exist so `build_quantum_context` and `run_job` keep reading
    `calib.ptc_by_name` / `calib.flat_by_name` across the split.
    """

    detector: DetectorCalibs
    filtered: FilterCalibs

    @property
    def physical_filter(self) -> str:
        return self.filtered.physical_filter

    @property
    def detector_ids(self) -> list:
        return self.detector.detector_ids

    @property
    def ptc_by_name(self) -> dict:
        return self.detector.ptc_by_name

    @property
    def linearizer_by_name(self) -> dict:
        return self.detector.linearizer_by_name

    @property
    def crosstalk_by_name(self) -> dict:
        return self.detector.crosstalk_by_name

    @property
    def flat_by_name(self) -> dict:
        return self.filtered.flat_by_name

    @property
    def intrinsic_zernikes_by_name(self) -> dict:
        return self.filtered.intrinsic_zernikes_by_name


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


def _build_detector_calibs() -> DetectorCalibs:
    """Load the filter-independent calibs fresh from disk. Always builds --
    `_ensure_detector_calibs` is where reuse is decided and where the load is
    timed, so this need not guard or time itself.

    Also where the detector-id list is established, since it comes from the
    ptc_*.fits filenames and so is itself filter-independent."""
    calibs = calib_dir()
    detector_ids = _discover_detector_ids(calibs)

    names_by_id: dict = {}
    ptc_by_name: dict = {}
    linearizer_by_name: dict = {}
    crosstalk_by_name: dict = {}

    for det_id in detector_ids:
        # Detector name comes off the calib object itself (._detectorName) --
        # no separate id->name table needed.
        ptc = ipIsr.PhotonTransferCurveDataset.readFits(
            os.path.join(calibs, f"ptc_{det_id}.fits")
        )
        name = ptc._detectorName
        names_by_id[det_id] = name
        ptc_by_name[name] = ptc
        linearizer_by_name[name] = ipIsr.Linearizer.readFits(
            os.path.join(calibs, f"linearizer_{det_id}.fits")
        )
        crosstalk_by_name[name] = ipIsr.CrosstalkCalib.readFits(
            os.path.join(calibs, f"crosstalk_{det_id}.fits")
        )

    return DetectorCalibs(
        detector_ids=detector_ids,
        names_by_id=names_by_id,
        ptc_by_name=ptc_by_name,
        linearizer_by_name=linearizer_by_name,
        crosstalk_by_name=crosstalk_by_name,
    )


def _build_filter_calibs(
    physical_filter: str, detector: DetectorCalibs
) -> FilterCalibs:
    """Load the filter-dependent calibs for `physical_filter` fresh from disk.

    Takes `detector` only for its `names_by_id`: the flats are named by id on
    disk but keyed by name in memory, and the ptcs are what establish that
    mapping."""
    calibs = calib_dir()
    flat_by_name: dict = {}
    intrinsic_zernikes_by_name: dict = {}

    for det_id in detector.detector_ids:
        name = detector.names_by_id[det_id]

        # Required.
        flat_path = os.path.join(calibs, f"flat_{det_id}_{physical_filter}.fits")
        if not os.path.exists(flat_path):
            raise RuntimeError(
                f"Missing required flat calib for detector {name} ({det_id}), "
                f"physical_filter {physical_filter!r}: {flat_path}"
            )
        flat_by_name[name] = afwImage.ExposureF.readFits(flat_path)

        # Optional (mirrors the real Butler connection's minimum=0): missing
        # file just means no entry for this detector.
        iz_path = os.path.join(calibs, f"intrinsicZernikes_{det_id}_{physical_filter}.fits")
        if os.path.exists(iz_path):
            intrinsic_zernikes_by_name[name] = ipIsr.IsrCalib.readFits(iz_path)

    return FilterCalibs(
        physical_filter=physical_filter,
        flat_by_name=flat_by_name,
        intrinsic_zernikes_by_name=intrinsic_zernikes_by_name,
    )


class CalibIds(NamedTuple):
    """What identifies each calib half, as the active backend sees it.

    This is the cache key for both halves, and it is the backend's answer rather
    than anything derived from the /prepare command: under the butler a
    recertification can change which datasets a given (filter, time) resolves to
    without the filter changing at all, so keying on the filter string would
    serve stale calibs indefinitely. Comparing identities *is* the staleness
    check, which is why the backends need no separate is-stale call.

    Both members must be hashable. Under the butler they are frozensets of
    dataset UUIDs; under files the filter half is the physical_filter string and
    the detector half is a constant, since a fixed calib_dir() cannot vary.
    """

    detector: Any
    filtered: Any


class FilesCalibBackend:
    """Calibs as flat FITS files under calib_dir(), the default backend.

    The testing path: needs no repo, no registry and nothing of the stack beyond
    afw/ip_isr. It cannot answer "which calib was valid at time T" -- the
    filename grammar has no slot for validity -- so `calib_time` is accepted and
    ignored. That is sound here because the grammar also admits only one calib
    per (type, detector[, physical_filter]), so there is never an ambiguity for a
    timestamp to resolve.
    """

    name = "files"

    def check(self) -> None:
        """Startup check: the calib directory exists and holds something.

        Only the directory, not the filenames: which calibs are required depends
        on the exposure's physical_filter, which no startup check knows. A
        missing ptc for one filter is a /prepare error; an empty or absent
        directory is a server that cannot serve any filter.
        """
        d = calib_dir()
        if not os.path.isdir(d):
            raise CalibConfigError(
                f"DONUT_SERVER_CALIB_DIR {d!r} is not a directory."
            )
        if not glob.glob(os.path.join(d, "*.fits")):
            raise CalibConfigError(
                f"DONUT_SERVER_CALIB_DIR {d!r} holds no .fits files; expected "
                "ptc_*.fits, linearizer_*.fits, flat_*.fits (see README)."
            )

    def resolve(self, command: dict) -> CalibIds:
        # The detector half is keyed on a constant: it is every ptc/linearizer/
        # crosstalk file in a directory that does not change under us, so there
        # is nothing for an identity to distinguish. Not None, which _ensure
        # uses for "nothing cached".
        return CalibIds(detector="files", filtered=command["physical_filter"])

    def load_detector(self, ids: CalibIds) -> "DetectorCalibs":
        return _build_detector_calibs()

    def load_filter(self, ids: CalibIds, detector: "DetectorCalibs") -> "FilterCalibs":
        return _build_filter_calibs(ids.filtered, detector)


class ButlerCalibBackend:
    """Calibs resolved through a real butler repo, as `pipetask run` would.

    The whole point is fidelity to the offline pipeline, so resolution is
    offloaded to the butler rather than reimplemented: the chained collection is
    flattened to its concrete children *in order* and passed whole, with the
    exposure's time as the timespan. Notably the chain is **not** filtered to its
    CALIBRATION children:

    - `AllDimensionsQuantumGraphBuilder` passes pipetask's `input_collections`
      unfiltered; there is no collection_types= anywhere on that path.
    - Rows from RUN/TAGGED collections get a synthetic unbounded validity range
      (`Timespan(None, None)`) which overlaps everything, so an uncertified
      dataset in a RUN satisfies any timespan.
    - Find-first ranks purely by position in the sequence, with no preference
      for CALIBRATION over RUN.

    So a producer run early in the chain legitimately shadows a certification
    later in it, and `LSSTCam/defaults` contains exactly such runs. That
    fallback looks like a bug and is not: it is what the pipeline would have
    used, which is the only answer this service is allowed to give. Filtering to
    CALIBRATION would make us resolve calibs the pipeline does not, and in
    /Users/jmeyers3/repo it resolves *nothing* at times when only the producer
    runs answer.

    Two traps, both deliberate:

    - `timespan` is always a real instant, never None. For a calibration dataset
      type `timespan=None` *excludes* the CALIBRATION collections rather than
      relaxing the constraint, so the obvious "no constraint" shortcut silently
      searches the wrong half of the chain.
    - `CalibrationLookupError` (two overlapping validity ranges in one
      CALIBRATION collection) propagates. Picking one would invent a policy the
      pipeline declines to invent, and `crosstalk` in the local repo really does
      have two open-ended spans -- so this is reachable, not theoretical.

    The Butler is constructed per resolve and closed, never cached: the task
    forks eight cutout workers per push, and an inherited live sqlite connection
    is unsupported even where it appears to work. Construction is a small
    fraction of the registry queries it enables, and resolution happens only on
    /prepare, so there is nothing to win by holding one.
    """

    name = "butler"

    # Both halves' dataset types, in the order DonutBlitzCornerConnections
    # declares them. intrinsicZernikes has minimum=0 on the connection, so a
    # miss is normal; the rest are required.
    DETECTOR_TYPES = ("ptc", "linearizer", "crosstalk")
    FILTER_TYPES = ("flat", "intrinsicZernikes")
    OPTIONAL_TYPES = frozenset({"intrinsicZernikes"})

    def check(self) -> None:
        """Startup check: the repo opens and the collections name something.

        Deliberately stops short of resolving a calib. That needs a
        physical_filter and a timespan, which belong to an exposure and not to a
        server -- and a repo that is missing one filter's flat is still a repo
        worth starting against. What this catches is the class of error where no
        request could ever succeed: an unopenable repo (the mistyped path), and
        a collection chain that flattens to nothing.

        The Butler is opened and closed here exactly as `resolve` does it, so
        nothing butler-shaped survives into the fork path; see the class
        docstring.
        """
        try:
            butler = Butler.from_config(butler_repo(), writeable=False)
        except Exception as exc:
            # The stack's own message already names the path and explains the
            # alias fallback, so it is quoted rather than replaced.
            raise CalibConfigError(
                f"DONUT_SERVER_BUTLER_REPO {butler_repo()!r} is not a usable "
                f"butler repo: {exc}"
            ) from exc
        try:
            # Raises its own message naming the repo when the chain is empty.
            _flatten_collections(butler, butler_collections())
        except CalibConfigError:
            raise
        except Exception as exc:
            raise CalibConfigError(
                f"DONUT_SERVER_BUTLER_COLLECTIONS {butler_collections()!r} "
                f"cannot be resolved in {butler_repo()!r}: {exc}"
            ) from exc
        finally:
            butler.close()

    def resolve(self, command: dict) -> CalibIds:
        """Resolve every calib to a DatasetRef and return the two UUID sets.

        Stashes the refs so the subsequent load_* calls need not resolve again;
        a cache hit discards them unused, which is the point of splitting
        resolve from load -- a hit must not pay to read pixels.
        """
        physical_filter = command["physical_filter"]
        timespan = _calib_timespan(command["calib_time"])

        butler = Butler.from_config(butler_repo(), writeable=False)
        try:
            collections = _flatten_collections(butler, butler_collections())
            detector_refs = self._find(
                butler, collections, timespan, self.DETECTOR_TYPES, None
            )
            filter_refs = self._find(
                butler, collections, timespan, self.FILTER_TYPES, physical_filter
            )
        finally:
            # Drops the registry's sqlite fd, so nothing butler-shaped survives
            # into the fork path.
            butler.close()

        self._detector_refs = detector_refs
        self._filter_refs = filter_refs
        self._physical_filter = physical_filter
        return CalibIds(
            detector=frozenset(ref.id for _, _, ref in detector_refs),
            filtered=frozenset(ref.id for _, _, ref in filter_refs),
        )

    def _find(self, butler, collections, timespan, dataset_types, physical_filter):
        """(dataset_type, detector_id, ref) for every calib that resolves.

        One find_dataset per (type, detector) rather than a bulk query: pipetask's
        own per-quantum path raises NotImplementedError for this task, because
        `detector` is not among its (instrument, visit) dimensions. A
        per-detector find_dataset bottoms out in the same SQL union and the same
        find-first window function, so it is equivalent where pipetask has no
        path at all.
        """
        found = []
        for dataset_type in dataset_types:
            for det_id in CORNER_DETECTORS:
                data_id = {"instrument": INSTRUMENT, "detector": det_id}
                if physical_filter is not None:
                    data_id["physical_filter"] = physical_filter
                ref = butler.find_dataset(
                    dataset_type, data_id, collections=collections, timespan=timespan
                )
                if ref is None:
                    if dataset_type in self.OPTIONAL_TYPES:
                        continue
                    raise RuntimeError(
                        f"No {dataset_type} calib resolved for detector "
                        f"{CORNER_DETECTOR_NAMES[det_id]} ({det_id})"
                        + (
                            f", physical_filter {physical_filter!r}"
                            if physical_filter is not None
                            else ""
                        )
                        + f" at {timespan} in collections {collections!r}."
                    )
                found.append((dataset_type, det_id, ref))
        return found

    def load_detector(self, ids: CalibIds) -> "DetectorCalibs":
        by_type = self._get_all(self._detector_refs)
        names_by_id = {d: CORNER_DETECTOR_NAMES[d] for d in sorted(CORNER_DETECTORS)}
        return DetectorCalibs(
            detector_ids=sorted(CORNER_DETECTORS),
            names_by_id=names_by_id,
            ptc_by_name=by_type["ptc"],
            linearizer_by_name=by_type["linearizer"],
            crosstalk_by_name=by_type["crosstalk"],
        )

    def load_filter(self, ids: CalibIds, detector: "DetectorCalibs") -> "FilterCalibs":
        by_type = self._get_all(self._filter_refs)
        return FilterCalibs(
            physical_filter=self._physical_filter,
            flat_by_name=by_type["flat"],
            intrinsic_zernikes_by_name=by_type.get("intrinsicZernikes", {}),
        )

    def _get_all(self, refs) -> dict:
        """{dataset_type: {detector_name: calib}} for already-resolved refs.

        Keyed by detector *name* to match the files backend and what
        build_quantum_context reads. The name comes from CORNER_DETECTOR_NAMES
        rather than the loaded object's `_detectorName`, so flats -- which are
        ExposureF and have no such attribute -- are keyed the same way as the
        ip_isr calibs.
        """
        out: dict = {}
        butler = Butler.from_config(butler_repo(), writeable=False)
        try:
            for dataset_type, det_id, ref in refs:
                name = CORNER_DETECTOR_NAMES[det_id]
                out.setdefault(dataset_type, {})[name] = butler.get(ref)
        finally:
            butler.close()
        return out


def _flatten_collections(butler, collections: list[str]) -> list[str]:
    """`collections` expanded to its concrete children, order preserved.

    Order is load-bearing: find-first ranks by position in this sequence and
    nothing else, so flattening must not sort or de-order. Chains are excluded
    from the result (they are replaced by their children) but a non-chain is
    returned as itself, so a bare RUN works too.
    """
    flat = [
        info.name
        for info in butler.collections.query_info(
            collections, flatten_chains=True, include_chains=False
        )
    ]
    if not flat:
        # CalibConfigError rather than a bare RuntimeError: an empty chain is a
        # property of the repo and the collection names, not of the request, so
        # when check() hits it at startup it should be fatal rather than retried.
        raise CalibConfigError(
            f"Collections {collections!r} resolved to nothing in "
            f"{butler_repo()!r}; is the repo the one you meant?"
        )
    return flat


def _calib_timespan(calib_time: str):
    """The /prepare `calib_time` string as a butler Timespan at that instant.

    Parsed here rather than in the front-end, which deliberately imports no
    astropy. TAI because that is what VisitInfo.date and the exposure records
    carry; a string with its own offset still parses correctly.

    An instant, not the visit's full timespan as pipetask passes. The two differ
    only when a validity boundary falls inside an exposure, where the range form
    raises CalibrationLookupError for an ambiguity an instant resolves cleanly --
    and an instant is the only thing the files path or a mid-exposure prepare
    could supply.
    """
    from astropy.time import Time

    try:
        when = Time(calib_time, scale="tai")
    except Exception as exc:
        raise CalibTimeError(
            f"calib_time {calib_time!r} is not a parseable time: {exc}"
        ) from exc
    # fromInstant rather than Timespan(when, when), which is empty and matches
    # nothing: a zero-width span excludes its own endpoints.
    return Timespan.fromInstant(when)


def calib_backend():
    """The configured calib backend, resolved per call like calib_dir().

    Per call rather than at import for the same reason: importing this module
    must need no data and no repo, and the tests set the variable per test.
    """
    kind = os.environ.get("DONUT_SERVER_CALIB_BACKEND") or "files"
    if kind == "files":
        return FilesCalibBackend()
    if kind == "butler":
        return ButlerCalibBackend()
    raise RuntimeError(
        f"DONUT_SERVER_CALIB_BACKEND is {kind!r}; valid values are 'files' "
        "(the default, flat FITS under DONUT_SERVER_CALIB_DIR) and 'butler' "
        "(resolved from DONUT_SERVER_BUTLER_REPO)."
    )


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
    # instead of failing, the same failure mode the filter cross-check guards.
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

    # physicalLabel, not bandLabel: band is a 6-valued coarsening of the filter,
    # so comparing it only checks a projection of the key. Two filters sharing a
    # band would pass that check while paired with the wrong flat -- the exact
    # failure this guard exists to catch. This compares the literal string that
    # selected the flat files.
    physical_filter = next(iter(exposures.values())).getFilter().physicalLabel
    prepared_filter = _CALIB_STORE.get("physical_filter")
    if prepared_filter != physical_filter:
        raise RuntimeError(
            f"raws are physical_filter {physical_filter!r} but prepare loaded "
            f"{prepared_filter!r} -- the flats would be wrong"
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


def check_calib_config() -> None:
    """Fail now if the configured calib backend could never serve a request.

    Called from coordinator_main before the hello, which is the earliest point
    that can do this at all: the check needs daf_butler, and the front-end
    deliberately imports no part of the stack (uvicorn.run imports the app
    in-process, so an import there is an import into the server). donutServer.py
    validates everything it can without the stack and says so explicitly; this
    is the other half.

    Before the hello specifically, because the hello's contract is "fully ready
    to serve". A bad repo caught here therefore never reaches the state where
    both flights are up and /health reports ready while every /prepare 500s --
    which is exactly how a mistyped --butler-repo used to present.

    Raising CalibConfigError rather than exiting: the front-end owns the
    decision about what a non-retryable bring-up failure does to the process,
    and it is the only side that can report it to whoever started the server.
    """
    try:
        backend = calib_backend()
    except RuntimeError as exc:
        # An unknown DONUT_SERVER_CALIB_BACKEND, or a backend whose required
        # variable is unset. Both are env typos with the same remedy as a bad
        # repo, and donutServer.py's startup banner already warns about the
        # first -- this is what makes the warning fatal instead of advisory.
        raise CalibConfigError(str(exc)) from exc
    backend.check()


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

    # Before the hello, and reported over the Pipe rather than raised: a bad
    # repo is the one startup failure the parent must be able to *explain*,
    # since it is an operator typo and the operator is sitting at the terminal
    # that started the server. Everything above here fails by propagating,
    # whose traceback is the right surface for a bug; this is not a bug.
    try:
        check_calib_config()
    except CalibConfigError as exc:
        conn.send({"ok": False, "event": "fatal", "error": str(exc)})
        # Explicit, rather than returning and letting the finally below run: the
        # shared block's owner is the front-end, and a clean exit here races the
        # parent's reap for no gain. The parent has the message already.
        conn.close()
        _SHM_VIEW.release()
        _SHM.close()
        os._exit(1)

    # Sent only once, and only after the BLAS assert, the shared block and the
    # calib config are all good -- so "hello" means fully ready to serve, not
    # merely spawned. Without it the parent cannot distinguish a healthy child
    # from one that is 15 s into importing afw + ts_wep, or one that is about to
    # fail the assert.
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
                    if isinstance(exc, ConfigOverrideError):
                        kind = "config_override"
                    elif isinstance(exc, CalibTimeError):
                        kind = "calib_time"
                    else:
                        kind = None
                    conn.send({"ok": False, "error": str(exc), "kind": kind})
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

    VISIT = 2026071300478  # the r_57 exposure
    source = client.resolve_from_files(client.raw_dir(), VISIT)
    physical_filter = source.physical_filter
    print(
        f"exposure -> visit={source.visit} physical_filter={physical_filter} "
        f"detectors={sorted(source.handles)}"
    )

    boresight_ra, boresight_dec, calib_time = client.read_prepare_args(source)
    print(
        f"boresight -> ra={boresight_ra:.4f} dec={boresight_dec:.4f} "
        f"calib_time={calib_time}"
    )

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
        "physical_filter": physical_filter,
        "boresight_ra": boresight_ra,
        "boresight_dec": boresight_dec,
        "calib_time": calib_time,
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
