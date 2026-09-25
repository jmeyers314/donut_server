# Rubin WF Estimation Service

A long-lived local service that runs the corner-sensor wavefront pipeline (`DonutBlitzCornerTask`) on
demand instead of as a batch quantum. It takes raw images for one exposure and runs ISR, blitz detection,
WCS refit, donut selection/cutout and Danish WF fitting across the 8 corner sensors in parallel, returning
a small result table. It calls the real `runQuantum()` against a hand-built in-memory Butler on real raws,
calibs and reference catalogs — no stand-ins, no disabled connections.

Dev hardware: MacBook Pro M3 Pro (36 GB); eventual dedicated Mac Studio/mini. Deployment may ultimately be
Linux (USDF/summit) — see Open questions.

## Workload shape (measured)

One exposure, 8 sensors, band r, dev laptop. **Measure steady state, not the first job:** the first
`runQuantum` in a fresh coordinator costs 7.5–8.9 s; jobs 2 onward settle at 6.1–6.7 s.

| | value |
|---|---|
| raw on the wire | 38.6 MB/sensor -> **308.5 MB/job** |
| result | **212 KB** parquet (63 donut rows x 58 columns) |
| `/push`, steady state | **~6.9–7.1 s** |
| `/prepare` | ~3.4–4.1 s cold, ~0.001 s reused |
| coordinator RSS | 2.7 GB after prepare, oscillating 4.1–5.9 GB per job |
| cadence | one job roughly every ~30 s |

`/push`: receive -> shared memory 0.16 s; parse layout 0.00005 s; decode -> 8 `ExposureF` 0.37 s; build
butler + Quantum + refs 0.002 s; **`runQuantum` ~6.3 s**; slim + serialize 0.03 s; `gc.collect` 0.16 s.
Inside the task: butler `get`s 0.2 s, refcat load (8x `loadPixelBox`) 0.1 s, cutout 2.6 s, Danish fit
3.1 s, `run()` 6.0 s.

## Architecture

```
producer ──HTTP──▶  FastAPI front-end  (async; NEVER forks; imports NO LSST code)
                         │  prepare / push / status / result / result-table
                         │  raw pixels ──▶ shared memory block (reused)
                         │  part layout only, over a Pipe ▼
                    Coordinator process  (long-lived, single-threaded; holds
                         │                _CALIB_STORE + _REFCAT_STORE + the task)
                         │  task.runQuantum() forks its own pools ▼
                    8 fork workers  (inherit calibs + tasks via copy-on-write)
```

The FastAPI process is async/threaded and must **never** fork (fork-from-threaded-process is unsafe,
especially on macOS), so it delegates all compute to the coordinator. The **coordinator** is the process
that forks; it stays single-threaded (BLAS thread counts set to 1 *before* numpy import) and holds the
reusable calibs, the refcat cache, the `DimensionUniverse` and the task. It forks no pools of its own —
**the task does its own two-stage forking internally** (cutout, then WF fit), sized by
`ExecutionResources(num_cores=8)`. The producer is off-machine in production, localhost for testing.

**Raw pixels never cross the Pipe.** A `multiprocessing.Pipe` is a Unix-domain socketpair with an **8 KB**
buffer on macOS; 308 MB through it takes **8.5 s** (33 MB/s, ~37,700 round-trips) versus 0.16 s over
loopback HTTP — the small buffer plus `Connection._recv` mallocing *all remaining bytes* every iteration,
neither of which is bandwidth or copying. So the front-end streams the request body into a **reusable
`SharedMemory` block** created once at startup and sends only a `list[protocol.PartLayout]` over the Pipe;
the coordinator opens the block once (`track=False`) and decodes in place. Boundary cost **9 ms**.

**Why fork / CoW.** The calib set is 558 MB and **4.87 s** to pickle, so 8 spawn workers would spend ~39 s
serializing. ISR does not mutate the shared calibs (0 of 40 objects changed across real jobs, by
pickled-state hash), and the shared `CalibSet` is protected by fork isolation as well as `copy=True` —
`runQuantum` does its `get()`s in the parent, ISR runs in the children. Peak system delta is **3.7 GB**
(`baseline + 8 × ~500 MB` of private scratch), affordable against 36 GB. CoW is real but **not** free:
**4–10 GB of CoW faults per job**, since children touching inherited Python objects increment refcounts and
every refcount write dirties an object-header page. Fork-safety is validated against the real task suite
(ISR, astrometry, detection, Danish, batoid): no aborts, no hangs, across repeated jobs.

## Job lifecycle

States: `PREPARED → RECEIVING → COMPUTING → DONE` (or `ERROR`).

1. **prepare** (exposure starting, no pixels): loads/refreshes calibs into `_CALIB_STORE` (ptc, linearizer,
   crosstalk per detector; flat and intrinsicZernikes per detector+band) and refcat shards into
   `_REFCAT_STORE`, both behind reuse guards. Also pays the one-time `DimensionUniverse` + task construction
   + pyarrow init. Raw-independent precompute belongs here.
2. **push** (pixels ready): blob streamed into shared memory, layout handed to the coordinator, which
   rebuilds exposures, builds the butler, and runs the task. Blocks until done.
3. **status** / **result** (JSON metadata, optional `?wait=N` long-poll) / **result table** (parquet).

## HTTP interface

All endpoints require `Authorization: Bearer <token>`, except `GET /health`, where it is optional so that a
supervisor's probe can reach it.

- `POST /prepare` — JSON `{band, boresight_ra, boresight_dec}`. Boresight is in degrees and
  **required** (400 if missing, non-numeric or non-finite — `json` parses a bare `NaN` token, so finiteness
  is checked in the web process): it is what lets refcat shards preload before pixels exist. Returns
  `job_id`, state, `timings: {calib: {...}, refcat: {...}}`; both sub-dicts carry the same keys whether the
  work was done or reused, so the shape never changes.
- `POST /push/{job_id}` — raw binary blob (below). Returns terminal state plus per-stage timings.
- `GET /status/{job_id}`; `GET /result/{job_id}?wait=N` — JSON `ready`, `state`, `timings`, `summary`
  (row/detector/group counts, dropped columns), `table_url`; stays JSON so it remains long-pollable.
- `GET /result/{job_id}/table` — `application/octet-stream`: `donutBlitzCornerResults` as parquet, serialized with
  daf_butler's own `astropy_to_arrow` so the bytes match a real `butler.put`, multidimensional Zernike
  columns and per-column units included.
- `GET /health` — **readiness**, and the status code is the verdict: 200 only when the coordinator is
  `READY`, 503 for starting / restarting / degraded. No token gets `{status, ready}`; a valid token adds
  `generation`, `spawns`, `restart_attempts`, `pid`, `child_uptime_s`, `primed`/`primed_args`,
  `degraded_reason`, job counts, and `last_loss`. `last_loss.exitcode` is the highest-value field there:
  `-11` is a segfault (a fork-safety regression, the thing the fork-safety validation exists to catch),
  `-9` is SIGKILL, which on a service that oscillates 4–6 GB should be read as macOS jetsam, and `1` is a
  Python exception with a traceback in the log. `Cache-Control: no-store`.
- `POST /admin/restart` — 202 plus the killed pid. Forces the coordinator down and lets the ordinary restart
  path replace it; the handle for wedged-but-alive, and the only way out of `DEGRADED`. It only signals, and
  works while `_lock` is held by a stuck reader.

Every coordinator-liveness failure is **503** with `Retry-After` and a `reason` of `coordinator_lost` (30 s),
`coordinator_restarting` (15 s) or `coordinator_degraded` (3600 s) — 503 rather than 502 even for a death,
because retry libraries treat 503 + `Retry-After` as "come back" while 502 is often taken as semi-permanent,
and what the producer needs is in `reason`. `/prepare` and `/push` success bodies carry `generation`, so a
producer's own logs can attribute a latency spike to a restart. A push whose coordinator dies also lands the
job in `ERROR`, so `/status` and `/result` agree with what the client just saw instead of long-polling a job
that can never finish.

**Result slimming.** The raw output table is 28 MB, dominated by three per-donut postage-stamp columns:
`stamp` 18.5 MB, `wf_img` 4.6 MB, `model_img` 4.6 MB. The other 58 columns — including `zk_deviation_*`
(n, 27) and `zk_intrinsic_*` (n, 67), in microns — are ~212 KB as parquet. `coordinator.IMAGE_COLUMNS`
names the dropped three and is guarded by a test, since an upstream rename would silently re-admit 28 MB.

## Wire format (push blob), version 2

Single binary POST body (`application/octet-stream`), **not** base64, **not** JSON: fixed header (magic
`DWFS`, version, part count, payload length), per-part descriptors (name, offset, count), then concatenated
part bytes — a generic `{part_name: bytes}` map. **Each sensor contributes two parts**,
`<detector_name>:img` and `<detector_name>:meta`, flat rather than nested so each 37 MB image lands as its
own slice needing no second split copy. Parts are keyed by *detector name* (e.g. `R00_SW0`), read off the
exposure, so the coordinator indexes `_CALIB_STORE`'s `*_by_name` dicts with no id->name table. `VERSION`
is 2; a v1 peer would silently misread v2 part names.

Each pair is produced by `exposure_codec`, which ships the float32 image verbatim plus a small pickle of
everything else — see its module docstring for why the metadata carrier is a 1x1 subimage. Encode
~13 ms/sensor, decode ~33 ms/sensor, image bit-identical. Pickling the whole `Exposure` is 4x smaller on
the wire but at 220 ms encode + 121 ms decode per sensor it loses on any link faster than ~0.6 Gbit/s,
with the decode on the critical path.

Recommended for production (not present): xxhash checksum in the header. Early size-cap rejection is present.

## Reference catalogs

`ref_cat/` holds 8192 files, 23 GB. **File `N.fits` is exactly HTM level-5 pixel `N`**, i.e. exactly the 16
level-7 children `N*16 … N*16+15` — verified empirically. Ids 8192–16383 are the complete level-5 range and
all are present, so **the local set covers the whole sky** and a shard can never be missing. Row+column cut
from the 880 GB level-7 original: rows `phot_g_mean` mag ≤ 18 (matching the task's `magLimit.maximum = 18`),
12 columns — `id`, `coord_ra`, `coord_dec`, `coord_raErr`, `coord_decErr`, `phot_g_mean_flux`,
`monster_ComCam_{u,g,r,i,z,y}_flux`.

The task's `refCat` connection declares `dimensions=("htm7",)`. `refcat_store.py` bridges that gap by
resharding to level 7 **in memory** at prepare; its module docstring records the two facts that force the
shape of the code. The payoff: level-5 shards handed to the loader directly cost **1268 ms on the push
critical path**, level-7 reshards **118 ms**, bit-identically. Level 8 buys 38 ms more but needs the
DatasetType declared `htm8` against a connection that says `htm7`, so level 7 wins. A one-time on-disk
reshard to 131k files was rejected: it still needs the astropy→afw conversion, gains only
off-critical-path prepare time that is cached anyway, and costs 131k inodes plus a second catalog copy.

The coordinator envelopes a `FIELD_RADIUS_DEG = 2.0` circle at level 5 about the boresight (10 shards,
101 MB, 1.15M rows for the r-band field), reads those files, reshards, and caches keyed on the *set* of
level-5 ids, so AOS dithers within one pointing reuse it (0.3 ms). Rotator angle is deliberately not in the
API: the region is a circle centred on the boresight, so rotation cannot change which shards it touches.
Max angular distance from boresight to any corner-sensor pixel, with the 400 px margin, is **1.878°**; a
test derives this from the raws rather than asserting the literal, so margin drift fails loudly.

Two non-obvious requirements, both of which fail confusingly:

1. **The refCat dataId must carry a dimension record**, or shard filtering on `dataId.region` surfaces as
   `AttributeError: 'NoneType' object has no attribute 'intersects'`. skypix regions are pure geometry, so
   the record needs no Registry — `universe["htm7"].RecordClass(...)` then `.expanded({...})`.
2. **Every shard catalog must share one `Schema` *object*.** Otherwise `loadRegion` raises `TypeError:
   Reference catalogs have mismatching schemas`, so `refcat_store.schema()` is a module singleton.

One benign warning per detector is expected: "Proper motion correction not available for this reference
catalog" — the column cut dropped `pm_ra`/`pm_dec`/`epoch` and `requireProperMotion` defaults to `False`.
Known limitation: the row cut was on `phot_g_mean` mag, so donut selection on `monster_ComCam_<band>` mags
is slightly incomplete near its own faint limit.

## Data on disk

`raw/raw_<visit>_<detectorId>_<band>.fits` — 48 files, 6 exposures x 8 detectors, one per band (u,g,r,i,z,y).
Detector ids are always `[191,192,195,196,199,200,203,204]` = `R00_SW0/SW1, R04_SW0/SW1, R40_SW0/SW1,
R44_SW0/SW1`. Each is an **un-assembled** `ExposureF`: bbox (0,0)–(4607,2047), image float32, mask int32
all-zero, variance all-zero, WCS present, Detector attached with 8 amps whose `rawBBox`es tile the image
exactly — overscan present, ISR-ready. `VisitInfo.id` matches the filename's visit; `DAYOBS`/`GROUPID`
supply `day_obs`/`group`.

`calib/` holds `ptc_<det>`, `linearizer_<det>`, `crosstalk_<det>` (band-independent) and
`flat_<det>_<band>`, `intrinsicZernikes_<det>_<band>` for all six bands.

## The hand-built Butler

No Registry, SQLite, or obs_lsst camera package. `coordinator.build_quantum_context` constructs a bare
`DimensionUniverse()`, one `DatasetType` per connection, a `DataCoordinate` per dataId, `DatasetRef`s, an
`InMemoryLimitedButler`, a `Quantum`, then `connections.buildDatasetRefs()` and a `QuantumContext`. Three
requirements fail confusingly when violated — implied dimension keys, a `Quantum.inputs` key for every
input even when empty, and the refCat dimension record — each documented as a comment at the line that
depends on it.

Cost is negligible: `put` of all 208 input datasets (48 + 160 refcat shards) takes **2 ms**,
`buildDatasetRefs` ~0. `put` is a bare dict insert; the deep copy happens on `get` (~7 ms/100 MB exposure,
0.46 s for the whole input set) because `InMemoryLimitedButler` hardcodes `copy=True`. **So nothing is
worth pre-building or pre-filling at prepare time** — except the refcat shards, whose whole point is to be
small enough that this deep copy is affordable. Patching `put` to `copy=False` saves ~0.6 s bit-identically
but costs +3 GB CoW, monkeypatching a pipe_base test class, and a safety property — deliberately not taken.

## Component map

The modules live in `python/lsst/ts/donut_server/` and are imported as
`from lsst.ts.donut_server import protocol`; they are named bare below for brevity. `tests/` stays at
the repo root, and the three data trees are located by `DONUT_SERVER_{CALIB,REFCAT,RAW}_DIR`.

- `protocol.py` — blob pack/unpack, `parse_layout` (framing only, no payload copies), `split_parts`.
  Stdlib-only on purpose, so it stays cheap for the web process.
- `exposure_codec.py` — `ExposureF` <-> wire bytes. Imported by `client.py` and `coordinator.py`, never by
  `protocol.py`.
- `refcat_store.py` — the level-5 → level-7 bridge: shared `Schema`, HTM geometry
  (`shard_ids_for_pointing`, `shard_ids_for_exposures`), `htm_data_id`, `load_and_reshard`, `RefCatStore`
  cache. Imported only by `coordinator.py`.
- `coordinator.py` — the engine: both stores, prepare-time loading, `reconstruct_exposures`,
  `build_quantum_context`, `run_job`, result slimming and parquet serialization, and the serial command loop
  `coordinator_main(conn, shm_name)`. Runnable standalone for a no-FastAPI smoke test.
- `server.py` — FastAPI app, per-job state, and `Coord` (owns the coordinator process, the Pipe and the
  shared block; serializes access behind asyncio locks, and owns coordinator liveness: death detection,
  bounded background restart, re-priming, and the terminal `DEGRADED` state).
- `client.py` — test producer: exposure discovery/selection, boresight read, encoding, parquet decode.
- `tests/` — protocol framing/layout, codec round trip, client discovery, result serialization and summary,
  refcat geometry/schema/reshard. Data-dependent parts skip when `DONUT_SERVER_RAW_DIR` or
  `DONUT_SERVER_REFCAT_DIR` is unset or empty.
  `test_coord_lifecycle.py` drives `Coord` against `tests/fake_coordinator.py` through its ctor seams
  (`target`, `target_args`, `shm_size`, `max_restart_attempts`, `backoff`, `on_restart`), so it imports no
  LSST code and needs no calibs. The fake's `target` must stay a module-level function — spawn pickles by
  reference, so it is its *module* that gets imported in the child. Its script applies to the first child
  only, or every replacement would reproduce the crash and no restart could be observed.

## Key invariants to preserve

- **FastAPI process never forks**, and imports **no LSST code**. `import coordinator` is lazy, inside
  `Coord.start()`; otherwise afw + ts_wep add ~15 s to uvicorn startup and ~1 GB of unused RSS. Spawn
  pickles the target by reference, so the child still imports it fine. The result parquet is held as opaque
  bytes for the same reason.
- **Coordinator single-threaded at fork time.** BLAS/OpenMP/Accelerate thread counts set to 1 before numpy
  import; workers likewise. `threading.active_count() == 1` is asserted at startup.
- **Do not reintroduce `gc.freeze()`.** It is a *once, before forking* call — frozen objects are permanently
  exempt from cyclic collection — and per-job use stranded ~1,101 objects every job with RSS still climbing
  after 7 (3.4 → 5.7 GB). It buys nothing measurable, since the refcount writes that actually dirty pages
  are unaffected by it. The explicit `gc.collect()` after each job stays (0.13 s to return ~900 MB). Watch
  the RSS *trend* and `gc.get_freeze_count()` (should be 0), not absolute RSS, which oscillates 4–6 GB.
- **The shared block is a single reusable buffer, so pushes must serialize.** `Coord.push_lock` wraps
  receive-and-dispatch; serializing only *coordinator* access would let two `PREPARED` jobs be received
  concurrently, which with one buffer is memory corruption.
- **`memoryview`s onto the shared block must be released before `close()`** or it raises `BufferError`. Each
  of the three view sites releases in a `finally`.
- **`track=False`** when the coordinator opens the block, or the child's `resource_tracker` unlinks it on
  exit. The front-end creates and is the sole unlinker.
- **The raws' band is cross-checked against the band `prepare` loaded** — otherwise the wrong flats are
  applied silently.
- **The raws' refcat shard coverage is cross-checked against what `prepare` loaded**, same reasoning: a
  pointing mismatch would degrade astrometry silently. The check compares level-5 id *sets* computed from
  the raws' own WCSs, not an angular tolerance, and **must use a 400 px margin, not 300** — see
  `TASK_PIXEL_MARGIN_PX + LOADER_BBOX_PADDING_PX` in `refcat_store`, where both terms and the reason each is
  needed are recorded, with a test pinning the second against `loadPixelBox`'s signature.
- **Every refcat shard catalog shares one `Schema` object**, and flux fields keep `units="nJy"`.
- **One job at a time.** The coordinator finishes one Pipe command before reading the next.
- **Coordinator-death detection must poll `proc.is_alive()`; no fd is trustworthy.** Measured: when the
  coordinator dies with even one fork worker still up, *neither* fd-based source reports it. `popen_fork`
  closes only the fds it creates, so the task's 8 workers inherit copies of both the spawn sentinel's write
  end and the Pipe itself — the sentinel stays unreadable and the Pipe never reaches EOF, so a plain
  `recv()` blocks **indefinitely** under `Coord._lock`, and every later request hangs behind it while the
  process stays up and stops answering. This is precisely the crash-during-`runQuantum` case, i.e. the
  likeliest crash. `waitpid(WNOHANG)` is the only source immune to descendants; `Coord._await_reply` polls
  it in 0.25 s slices. The parent's own `child_conn.close()` is *not* what saves this — refcounting already
  closed that local, verified — so do not "simplify" the poll loop back to a bare `recv()` on its strength.
- **Reap the process *group*, not the process.** `coordinator_main` does `os.setpgid(0, 0)` so `Coord._reap`
  can `killpg` the coordinator together with its fork workers; orphans would otherwise survive holding the
  shared-block mapping and 8 cores. `_signal_child` tries `killpg` first and unconditionally, which is safe
  because a group id equal to the child's pid can never be the front-end's own — and it still reaches
  stragglers after the coordinator itself has been reaped. Consequence of that `setpgid`: the coordinator no
  longer receives the terminal's SIGINT. Ctrl-C still works, because uvicorn's handler runs lifespan
  shutdown → `coord.aclose()`, and front-end death still closes the last `parent_conn` so the child exits on
  `EOFError`.
- **The reader thread lives in a dedicated executor.** uvicorn's shutdown calls
  `loop.shutdown_default_executor(300)`, so a reader parked in the *default* executor would make SIGTERM
  take five minutes. For the same reason `_await_reply` must never block unboundedly, and the exchange is
  never wrapped in `asyncio.wait_for`: cancelling the await does not stop the thread, so `_lock` would be
  released while a live thread still owned `self._conn`, and a late reply could be handed to the *next* job.
- **`/health`'s status code carries the verdict** — 200 only when `READY`, 503 otherwise. Nothing can
  supervise an endpoint that always returns 200, which is what it used to do. Auth is optional there, not
  absent: a bare probe has no bearer token. `is_alive()` alone is unusable as readiness — it is `True` for a
  child 15 s into importing afw + ts_wep, and for one whose fork pool is deadlocked.
- **Auto-re-priming is an availability optimization, not a correctness requirement.** A restarted-but-
  unprimed coordinator fails the next push loudly via the band and refcat-coverage cross-checks above, and
  that is what makes replaying the last `prepare` safe to do automatically. Skipped when the crash was
  provoked *by* a `prepare`, to avoid a crash loop.

## Not yet built

- **launchd supervision** — plist wrapping `uvicorn` with `KeepAlive`, absolute paths, Agg-only matplotlib.
  Covers front-end death and reboot only, which is orthogonal to the coordinator restart above. Needs
  `KeepAlive: true`: uvicorn exits **0** on SIGTERM, so `SuccessfulExit: false` would decline to restart it.
  The front-end deliberately does **not** self-terminate — launchd cannot probe HTTP, so the only way to
  involve it is to exit, and reaching `DEGRADED` almost always means a *deterministic* failure that a full
  recycle will not fix, converting a `Coord`-level crash loop into a slower launchd-level one while losing
  the `/health` explanation.
- **Automatic detection of a wedged-but-alive coordinator.** `is_alive()` is `True` for a child whose fork
  pool is deadlocked, and there is deliberately no command timeout: a generous bound cannot be sized safely
  against a 6.1–8.9 s push, and a mis-sized one kills healthy work. `POST /admin/restart` is the intended
  handle — an operator or external watchdog that knows a job has been `COMPUTING` for minutes has strictly
  more information than a blind timeout.
  **Auth/TLS hardening** for off-machine producers.
- **Result TTL / reaper.** `JOBS` grows unbounded; each entry holds ~212 KB of parquet.
- **Non-blocking push.** The state machine already supports `COMPUTING → DONE` if the call moves to a
  background task.
- **Persistence.** Both stores and `JOBS` are in-memory; a crash loses them.

## Optimizations to revisit

- **The task's own internals** — at ~6.3 s of a ~7.0 s push (cutout 2.6 s, Danish 3.1 s), this is where the
  remaining budget is. Nothing in the wrapper is worth optimizing by comparison.
- **First-job warm-up** — 7.5–8.9 s against 6.3 s steady state, and nothing in `prepare` pays it down.
  Hoisting it is worth ~2 s on the first exposure after any restart.
- **Overlapping next-job RECEIVING with current-job COMPUTING** is *foreclosed* by the single shared buffer;
  it needs two blocks (or a ring) plus revisiting `push_lock`. At 0.16 s of receive against ~6.3 s of
  compute, unnecessary.
- **Network:** wired Ethernet mandatory — 308 MB is ~2.7 s on 1 GbE, ~0.28 s on 10 GbE. Widening the Pipe's
  `SO_SNDBUF`/`SO_RCVBUF` is cheap hardening, since prepare responses still cross it.

## Open questions

1. **Deployment target: Mac or Linux?** On Linux CoW works cleanly and is measurable, and the shared-memory
   win is smaller (pipes default to 64 KB and `AF_UNIX` autotunes, so the 8 KB cliff largely disappears —
   design still right, payoff less dramatic). Fork-safety of the real task suite and the fork/CoW/memory
   numbers should be re-validated there early: page size, allocator and fork behaviour all differ.
2. **Can `runQuantum` get under ~5 s?** Currently ~6.3 s on the dev laptop with 8 cores, 0.1 s of it refcat
   loading. Needs a decision on whether the target is real, the hardware changes, or the task gets faster.

## Verification

See README.md for the eups setup and the three `DONUT_SERVER_*_DIR` variables this assumes.

```zsh
python -m pytest tests/ -q                      # 85 tests
python -m lsst.ts.donut_server.coordinator      # full prepare -> push, no FastAPI
bin/donutServer.py                              # then, in another shell:
bin/donutClient.py --token <tok> --visit 2026071300478 --wait 60
```

The client prints the boresight, per-stage timings, the result summary, and real Zernikes
(`zk_deviation_ccs`, shape (n, 27), microns). For the r-band exposure expect boresight
`ra=283.6660 dec=-28.1326`; prepare reporting `n_level5_files: 10`, `n_level7_shards: 160`,
`n_rows: 1148693` in ~0.5–0.7 s cold and ~0.0003 s on a repeat; `n_input_datasets: 208`; and **63 donut
rows** across 8 detectors with 28/29 groups fit. `donut_id` values are Gaia source ids (e.g.
`6761235373898405888`) — the quickest confirmation the refcat path ran rather than a blind fallback, as is
the *absence* of the "No reference catalog shards provided" warning, given that the "Proper motion
correction not available" warnings at the same level do appear (8 per job, expected).
