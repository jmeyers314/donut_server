# donut_server

A long-lived local service that runs the Rubin corner-sensor wavefront pipeline
(`DonutBlitzCornerTask`) on demand instead of as a batch quantum. It calls the real `runQuantum()`
against a hand-built in-memory Butler on real raws, calibs and reference catalogs.

See [Rubin Wavefront Estimation Service.md](Rubin%20Wavefront%20Estimation%20Service.md) for the
architecture, the measured workload shape, and the open questions.

## Environment

The service imports `lsst.afw`, `lsst.ip.isr`, `lsst.daf.butler`, `lsst.pipe.base` and
`lsst.ts.wep`, so an eups-setup stack is a hard requirement — there is no standalone install:

```zsh
source /Users/jmeyers3/src/lsstinstall/loadLSST.zsh
setup lsst_distrib
setup -kr ~/src/ts_wep      # required, and must come first -- see below
setup -kr .
scons                       # builds bin/ from bin.src/; required for the launchers
```

`ts_wep` is set up from a checkout deliberately: the service tracks unreleased blitz-task changes.
It is declared `setupRequired`, so if it is not already set up, `setup -kr .` stops with
`Product ts_wep not found` — set up `ts_wep` first and re-run.

`scons` is not optional if you want the launchers: `bin/` is a build product and a fresh checkout has
none. On macOS, SIP strips `DYLD_LIBRARY_PATH` from any child of a system-volume binary — which
includes `/bin/sh` and `/usr/bin/env`, even named by absolute path — so `import lsst.afw.image` would
fail with `Library not loaded: libbase.dylib`. `scons` rewrites the `#!/usr/bin/env python` shebang in
`bin.src/` to the absolute conda interpreter, which is not SIP-protected, and that is what makes the
built launchers work. Running the `bin.src/` copies directly will fail for exactly this reason.

`LD_LIBRARY_PATH` does survive SIP. If you wrap these in a shell script of your own, re-export it:

```zsh
export DYLD_LIBRARY_PATH="$LD_LIBRARY_PATH"
```

### Shared memory

On Linux the server needs **513 MB of `/dev/shm` per flight** — 1 GB at the default
`--num-flights 2`. Raws reach the coordinator through a shared block rather than the Pipe, one
reusable block per flight, sized for the largest push the service accepts.

```zsh
df -h /dev/shm      # Avail must exceed 513 MB x flights
```

A container is the usual problem: Docker, Podman and Jupyter images default `/dev/shm` to 64 MB,
which is not enough for even one flight. Start it with `--shm-size=2g`, or run fewer flights.
`FlightPool.start()` checks this and refuses to start with a message naming both numbers, because
the runtime failure it replaces is undiagnosable: tmpfs reserves nothing at allocation time, so an
oversubscribed block is only discovered when a push writes a page that cannot be backed, and the
kernel's only recourse then is `SIGBUS`. That kills the front-end mid-request — the producer sees a
closed socket with no HTTP status, and the server prints `Bus error (core dumped)` and nothing else.

Note that tmpfs pages count against a cgroup memory limit, so a container can have a large
`/dev/shm` and still `SIGBUS`; the startup check cannot see that case. macOS has no `/dev/shm` and
its POSIX shared memory is kernel-backed with no mount to measure, so the check is skipped there.

## Data

Not in the repo (33 GB, git-ignored). Locate it yourself — each directory is found through its own
environment variable, so the data need not live in the checkout:

| variable | needed by | size | contents |
|---|---|---|---|
| `DONUT_SERVER_RAW_DIR` | **client** | 5.1 G | real corner-sensor raws, overscan present, ISR-ready |
| `DONUT_SERVER_CALIB_DIR` | server | 4.8 G | ptc/linearizer/crosstalk per detector; flats and intrinsic Zernikes per physical_filter |
| `DONUT_SERVER_REFCAT_DIR` | server | 23 G | Gaia level-5 shards, resharded to level 7 at prepare time |

`DONUT_SERVER_RAW_DIR` is read by `client.py` alone (and `--raw-dir` overrides
it), and holds *assembled* raws. The server never opens one of those: they
arrive over shared memory on `/push`, so a server host needs neither the
variable nor the 5.1 G.

It does open DAQ-native amplifier files, on `/ingest` and `/submit` — the
server-side ingest path, where a producer sends a directory instead of 294 MiB
of pixels. That directory has to be visible to the server, and
`DONUT_SERVER_REPLAY_DIR` has to be set there too if the exposures are BTS
replays, since the substitution happens server-side on that path. `/push` is
unchanged and remains the only option for a producer on another host.

`DONUT_SERVER_STAMP_DIR` is the one output directory, and it is required too. It needs no data up
front, only somewhere writable: the coordinator puts one `<job_id>.parquet` there per job, holding
the full result table including the per-donut stamps, plus two smaller tables and a `plots/`
subdirectory of rendered PNGs — see [Diagnostic plots](#diagnostic-plots).

```zsh
export DONUT_SERVER_CALIB_DIR=$PWD/calib
export DONUT_SERVER_REFCAT_DIR=$PWD/ref_cat
export DONUT_SERVER_RAW_DIR=$PWD/raw
export DONUT_SERVER_STAMP_DIR=$PWD/stamps
```

Unset is a loud `RuntimeError` naming the variable, not a silent empty result — but only in the
process that needs it, and only when it needs it. A server with no `DONUT_SERVER_RAW_DIR` starts and
serves jobs normally; one with no `DONUT_SERVER_CALIB_DIR` starts too, and fails at the first
`/prepare` (unless `-b` put it on the butler backend, which needs no calib tree). The
data-dependent tests skip when unset, so `scons`/`pytest` still pass without the data — check the
skip count.

### Calibs from a butler repo instead

The calib loader is a backend, because the flat tree above cannot answer the question production
needs answered: *which* calib was valid when this exposure was taken. Filenames carry no validity
information, so the files path can only return whatever is on disk.

**Naming a repo is what selects the butler backend**; with no repo, calibs come
from the tree above:

```zsh
bin/donutServer.py -b LSSTCam                           # butler-resolved calibs
bin/donutServer.py                                      # the flat FITS tree
```

| variable | flag | default | meaning |
|---|---|---|---|
| `DONUT_SERVER_BUTLER_REPO` | `-b`, `--butler-repo` | — | repo root or alias; giving it implies `=butler` |
| `DONUT_SERVER_BUTLER_COLLECTIONS` | `-i`, `--butler-collections` | `LSSTCam/defaults` | comma-separated, highest priority first |
| `DONUT_SERVER_CALIB_BACKEND` | — | `files` | `files` or `butler`; derived from the above, rarely set by hand |

`-b`/`-i` are the short forms `donutClient.py` and `pipetask run` use for the
same two things; `-j`/`-n` follow `pipetask run` too, for the process count and
the cores one quantum may use. There is no `--calib-backend` flag: the repo is
the only input the butler backend needs that the files one cannot use, so it
carries the choice. `DONUT_SERVER_CALIB_BACKEND` remains the backend's own
contract — the launcher derives it, and exporting it is how you pin a backend
the flags cannot express.

**A flag is a way of setting its variable**, which is the only transport that
reaches the coordinator children, so precedence is the usual
**flag > exported > default**. Passing `-b` therefore selects the butler backend
even against an exported `DONUT_SERVER_CALIB_BACKEND=files`; a repo that arrives
through the *environment* only defaults the backend, so an exported `=files` can
still pin the files path while a repo stays exported for another run.

Two shapes are argparse errors before the port is bound, rather than a server
that comes up `ready` and then 500s on every prepare: collections with no repo
(they would be silently ignored), and `=butler` with no repo to resolve from.
Each error names whichever source it came from, flag or variable. That the repo
*resolves* is not checked there — it would mean importing the stack into the web
process — so the coordinator checks it at the first resolve. The startup banner
names the calib source either way.

`DONUT_SERVER_CALIB_DIR` is required only under `files`, which is why it stays the default: the tests
and a laptop checkout need no repo, no registry and nothing of the stack beyond `afw`/`ip_isr`.

Under `butler`, resolution is **offloaded to the butler** so it matches what `pipetask run` would
have resolved for the same exposure and collection. The chain is flattened in order and passed
*unfiltered*, with the exposure time as the timespan. One consequence looks like a bug and is not:
a producer RUN early in the chain can satisfy a lookup that no CALIBRATION collection covers, because
RUN rows get a synthetic unbounded validity range and find-first ranks purely by position. That is
pipetask's own behaviour — filtering the chain to its CALIBRATION children would make this service
resolve *nothing* at times the pipeline resolves fine.

Both backends agree on the local data: same detectors, same `intrinsicZernikes` coverage, flats
identical pixel-for-pixel. If they ever diverge, one of the two trees is stale.

**These files are never pruned, and at a 30 s cadence they arrive at ~2.0 GB/hour** (16.7 MB a job,
measured on the r_57 exposure below — the in-memory columns are ~21 MB and parquet compresses). With
plotting on, the two extra tables and six PNGs take a job from ~18 MB to **~75 MB, so ~8.9 GB/hour**
— see [Diagnostic plots](#diagnostic-plots), which is also where to look if that is too much.
Deleting old
ones is an operator or cron job; nothing in the service does it. They are also safe to delete at any
time — the Zernikes a client waits for are served from the front-end's memory and never read back
off disk, so losing a stamp file costs only the stamps. A stray `.parquet.tmp` is the debris of a
coordinator killed mid-write and can go too.

### Why the stamps are a separate endpoint

The `/push` reply and `/result/<job_id>/table` carry 59 columns and ~214 KB; the stamps are the other
three columns and ~21 MB in memory (`stamp` is 167×167 a donut, `wf_img` and `model_img` 83×83).
Serializing and writing them costs ~0.1 s, which would otherwise land on the path a client is blocked
on, so the coordinator answers first and writes them in the ~20 s before the next exposure. Two
consequences worth knowing:

- `GET /result/<job_id>/images` answers **409 for a moment after the job reports DONE**. That is the
  write still in flight, not a failure — retry. `bin/donutClient.py --images` already does.
- A write that genuinely fails is logged and otherwise ignored, because by then the client has its
  Zernikes and a 200. `grep 'deferred image write failed' $DONUT_SERVER_LOG` is how you find out.

## Config overrides (`-c` / `-C`)

The blitz task's config is not hardcoded: `/prepare` accepts an ordered override list, and the client
exposes it with `pipetask run`'s own flags and semantics — because it is `pipetask`'s own
implementation (`lsst.pipe.base.configOverrides.ConfigOverrides`) doing the work.

```zsh
bin/donutClient.py --visit 2026071300478 -c maxFitScatter=2.0 -c donutSelector.magMax=16
bin/donutClient.py --visit 2026071300478 -C my_overrides.py          # a pipetask-style config file
bin/donutClient.py --visit 2026071300478 -C base.py -c savePlots=False   # later wins
```

`-c` and `-C` interleave and apply in **command-line order**, last write wins. `-C` sends the file's
*contents*, not its path, so the file need not exist on the server — which is also why it is the
client that reports an unreadable one. Both are validated during `parse_args`, before any raws are
read.

Overrides ride on `/prepare` rather than a separate endpoint so they survive a coordinator restart:
the front-end replays the last successful prepare verbatim, override list included. Consequently
**a prepare states the config in full** — a bare `/prepare` with no `-c`/`-C` resets to defaults, and
is how you back out of a bad override set.

Two ways to see what is actually installed:

- `GET /config` — the full `DonutBlitzCornerConfig` as loadable Python (~77 KB), from the last
  prepare. Cached by the front-end, so it answers even while the coordinator is restarting or
  degraded; `X-Donut-Stale: 1` says the dump may not describe the child running right now. The
  dashboard links to it.
- The `/prepare` reply and `/health` echo a **digest** of the list: `-c` entries whole, `-C` entries
  as a name plus line count and hash. The bodies are deliberately never echoed — `/health` is polled
  once a second.

A repeat prepare with an unchanged list reports `task: reused: true` and rebuilds nothing. The task,
calib and refcat guards are independent, so changing an override does **not** trigger a calib reload.
The calib guard is itself split: a *filter* change reloads only the flats and intrinsic Zernikes
(`filter_reused: false`), while the PTCs, linearizers and crosstalk — dimensioned by detector alone —
are loaded once per process and stay (`detector_reused: true`).

Each half is keyed on the *identity* the backend reports for it, not on the filter: under `butler`
that is the set of resolved dataset ids, so a recertification invalidates the half it actually
changed. Every prepare re-resolves (cheap, and reported as `resolve_s`) because the resolve is what
produces the key — but a hit then reads no pixels.

### Three sharp edges

- **`-C` is arbitrary code execution** in the coordinator, pre-fork, as the service user — `exec` is
  how pipetask config files work. Note this composes with the loopback exemption below: anyone who
  can reach 127.0.0.1 can run code here without a token. That is a widening of the existing
  concession that a shell user on this host is already trusted, from "can spend your CPU" to "can run
  code", so weigh it before exposing the port.
- **`-c savePlots=True` is a 400 at prepare**, and wanting it is a sign you want the plot flights
  below instead. Since ts_wep `f5ce1351` the three diagnostic plots are butler `Output` datasets, so
  turning this on leaves three output connections this service builds no refs for — the same
  rejection `-c doZernikesOutput=True` gets. The plots are drawn anyway, off the critical path; see
  [Diagnostic plots](#diagnostic-plots).
- **An override can be rejected.** Anything that perturbs the task's *connection set* (notably
  `-c doZernikesOutput=True`, which adds a second output this service builds no ref for) is a **400
  at prepare**, not a mysterious `KeyError` mid-push. Likewise a positive `hangTimeout` at or below
  `unitTimeout`, which would make the watchdog abort the coordinator on every ordinary job. In both
  cases the previously good config keeps serving.

`-c doSelectionOutput=False` is accepted and costs you one plot. The service flips that field's
upstream default to True, because the focal-plane selection plot needs the two tables it governs and
"persisting" them here means a dict insert into the in-memory butler. Turning it back off wires
`cornerResults` alone, and the focal-plane plot is then skipped while the other two still draw.

## Producers

Three ways to drive the service, in order of how production-like they are:

| launcher | picks the exposure | reads | for |
|---|---|---|---|
| `bin/donutClient.py` | `--visit`, from `raw_*.fits` or a butler repo | assembled raws | development, `--loop` cadence tests |
| `scripts/run_sequence.py` | walks a tree, one dir per exposure, at `--delay` | DAQ-native amp files | replaying a night at a chosen cadence |
| `bin/donutTrigger.py` | one directory, given to it by the camera | DAQ-native amp files | **production and BTS** |

The last two share the DAQ read path in `daq.py`: the camera writes *unassembled amplifier* images,
one FITS per corner sensor with eight `Segment1x` HDUs, and the wire format carries an assembled
`ExposureF`, so assembly happens producer-side until the raws are ingested into a butler.

### The camera trigger

In production the camera drives the service. Whenever it writes a FITS file it invokes a per-file
script that compresses it, writes a JSON sidecar, and ships it to OODS and USDF; one extra step in
that script calls `bin/donutTrigger.py` with the exposure's directory. The hook is wired for **one
sensor only** — `R00_SW0` — so a script that runs per *file* fires once per *exposure*:

```zsh
bin/donutTrigger.py /path/to/MC_O_20260713_000050            # poll, hand the directory over
bin/donutTrigger.py /path/to/exposure --dry-run              # resolve and report, submit nothing
bin/donutTrigger.py /path/to/exposure --detach --log-dir /var/log/aos
```

Firing on the first corner sensor means the other seven may not have landed, so it polls until all 8
`_R??_SW?.fits` have sidecars *and* have stopped growing — every size unchanged across
`--settle-time` seconds, since a 9.5 MB FITS that exists is not necessarily one that is finished.
Only then does it POST the *directory* to `/submit`, which resolves the headers, prepares against
them, and reads the pixels itself.

The two polling knobs are deliberately independent:

- **`--poll-interval`** (default 0.25 s) is sampling granularity, and nearly free — a poll is one
  glob plus up to 16 stats, measured at 0.6 ms median / 2.6 ms worst case on USDF. It bounds only
  how late a finished exposure is *noticed*, and that lateness **is** charged to the reported time.
- **`--settle-time`** (default 1.0 s) is the safety property: how long a stalled write may pause
  without being mistaken for a finished file. It is enforced against the clock, so it holds at
  whatever the interval is.

Spelling the window as a poll *count* (the old `--stable-checks`) coupled the two, so lowering the
interval for latency silently shortened the guard. The settle window is reported separately (`+1.0s
waiting for sizes to settle`) and excluded from the times the trigger prints, because it watches a
directory that already stopped changing — charging it to the exposure would make every number
~`settle_time` pessimistic and hide a real regression that size.

**Nothing is prepared while the shutter is open**: everything happens after
readout, which costs the latency the two-phase protocol was built to avoid, but the camera hook is
the only trigger available today.

**The trigger imports no part of the stack**, which is the point: it polls and POSTs, so the whole
process costs 0.21 s against 2.77 s for a producer that resolves its own headers (measured on this
dev box — the header stack is ~3 s and afw ~1.9 s, and they are additive). The server also keeps its
camera and header-translator caches warm across exposures, which a fresh producer process cannot.

The trade is that the server must see the directory — and must see it *before* the camera's `fpack`
step, since `readRawAmps` cannot read a `.fz`. A producer on another host, or one holding assembled
exposures or a butler, uses `donutClient.py` and the `/prepare` → `/push` path instead; that is also
where `--images` and `--wait` live, both needing the stack to interpret a result.

**`--detach` still matters.** The import is gone but the *wait* is not: `/submit` covers the whole
fit, so the notifier blocks ~25 s on it (measured). With `--detach` the parent returns in 0.38 s and
the child waits, which is the point for a hook on the camera's own critical path. The watchdog
remains for the same reason — nothing supervises that child.

Two guards that used to live in the producer are now `/submit`'s, because both need the visit or the
image type and a notifier has neither: `DARK`/`BIAS`/`FLAT` come back `SKIPPED` without spending a
flight (`--any-imgtype` forces them through), and a visit already submitted comes back 409
(`--resubmit` runs it again, which is what the trigger's `--state-dir ''` was for). `--dry-run`
resolves and reports without preparing — the check to run first on a new deployment.

`--detach` is for the hook, which sits on the camera's own critical path: it re-runs the command in a
new session with output to `--log-dir/<exposure>.log` and returns in ~1.7 s, against a 0.73 s
bare-interpreter baseline — i.e. what is left is Python startup. That is only true because
`trigger.py` imports `client` *lazily*; it reaches `lsst.afw.image` through `exposure_codec`, and a
top-level import would silently put ~5 s back on the hook. A test asserts the module stays clean of
`lsst.afw` for that reason. The trade is that the child's exit status reaches nobody, so a failure is
visible only in the log — which is why the detached child also arms a `SIGALRM` watchdog over its
whole life rather than risking an orphan.

Two guards worth knowing, because both are silent when they work:

- **A hook that fires twice submits once.** Before pushing, the trigger claims
  `<state-dir>/<visit>.done` with `O_CREAT | O_EXCL`; a second firing logs and exits 0. Keyed on the
  *visit*, not the directory name, so the same pixels replayed twice are still caught. `--state-dir ''`
  disables it.
- **Calibration frames are skipped.** `DARK`/`BIAS`/`FLAT` exit 0 without pushing, since they have no
  stars and would burn a flight to reach an empty donut catalog. `--any-imgtype` forces them through.
  Anything outside `CWFS`/`ACQ`/`SCIENCE`/`OBJECT` is processed but flagged.

Note the trigger reads **uncompressed** DAQ files: a directory holding only `.fz` files fails naming
that as the cause, because `readRawAmps` cannot read one and whether our step runs before or after
the camera script's compression step is still an open question.

### Replayed exposures at BTS

The Base Test Stand replays real images through the simulated camera, so files appear exactly as they
would on the real camera — but **BTS replays the pixels, not the metadata**. The headers carry the
simulated telescope's time and pointing, not the values present when the data was taken.

**The service cannot detect this, and will not refuse it.** This was measured, and the result is the
reason substitution exists rather than being a convenience. Pushing a replayed exposure under BTS's
own headers — `FILTER z_20`, boresight `(120, -45)`, a date three months off, against pixels actually
taken in `r_57` at `(215.5, -10.1)` — came back `DONE`, with 74 donut rows and 6 of 6 groups
"succeeded". Every cross-check passed, because each one checks *consistency*, not truth:

- the filter check compares the pushed raws against what `/prepare` loaded, and both said `z_20` —
  consistently wrong, so the `z_20` flat was paired with `r_57` pixels without complaint
- the refcat-coverage check asks whether the shards loaded cover the raws' WCS, and that WCS was
  built *from the same bad boresight*, so the shards covered it exactly. Donuts in pixels from one
  field were matched against Gaia sources from another, and the fit converged on that

So the answer looks healthy and is meaningless. Nothing downstream flags it. With substitution the
same pixels give `visit=2026071300050 physical_filter=r_57`, 62 rows and 8 of 8 groups, with
`donut_id` values that are genuine Gaia ids for the field the pixels are of.

| variable | meaning |
|---|---|
| `DONUT_SERVER_REPLAY_DIR` | a replay tree, one subdirectory per original exposure, named by OBSID. **Setting it switches substitution on**; unset, headers come from the polled directory, which is what the real camera needs |
| `DONUT_SERVER_REPLAY_OBSID` | pin one exposure and skip the lookup — for hand testing, and for a replay whose marker is missing or names something not in the tree |

```zsh
export DONUT_SERVER_REPLAY_DIR=/path/to/replays
bin/donutTrigger.py /path/to/live/exposure --dry-run
```

Which original is being replayed is read from **`EMUIMAGE`**, the emulator's own record of what it
replayed. Its value is a bare OBSID, matching how the replay tree names its exposure directories, so
the lookup is a path join:

```json
"OBSID"    : "MC_O_20261022_000001",     // what BTS wrote
"EMUIMAGE" : "MC_O_20250530_000395",     // what it replayed
```

It is read from **all 8 sensors and cross-checked**, not trusted from one: a directory holding files
from two different replays would otherwise substitute successfully, pass every downstream check, and
answer wrongly. Disagreement names each value and the detectors that carried it.

The keyword is absent on real camera files (verified across a tree of originals, including a check
for blank-valued cards), which is what makes it safe to *require*. With `DONUT_SERVER_REPLAY_DIR`
set, an absent or blank `EMUIMAGE` is a hard error naming the offending detectors — that is what
pointing the replay path at real data looks like, and continuing would submit it under whatever
pointing happened to be in the headers.

**Setting the variable is the switch, not the marker's presence.** Running a replay *without*
substitution is a deliberate test — forcing every WCS to describe the wrong sky — so unsetting the
variable has to keep working, and it does.

What gets substituted is the **whole primary header**, per detector, so filter, pointing, times and
`FOCUSZ` all come from the original observation and stay consistent with the pixels. Two rules make
that safe, and both are load-bearing: the substitute is matched to a live file by **detector name read
from the live header** (`RAFTBAY` + `CCDSLOT`), never by filename, and the detector *number* likewise
comes from the live file, because it indexes cameraGeom for the amp layout the pixels are actually in.
Only metadata describing the *observation* is taken from the substitute.

Every replay logs what it matched and what it changed, so a desynced replay is visible in the
trigger's own log rather than only downstream as an inexplicably bad fit:

```
replay: EMUIMAGE names original exposure MC_O_20260713_000050
replay: EMUIMAGE MC_O_20260713_000050 -> loaded MC_O_20260713_000050
replay metadata substitution (live -> substituted):
    OBSID     BT_S_20261023_000007    -> MC_O_20260713_000050   *
    FILTER    g_6                     -> r_57   *
    RA        120.0                   -> 215.53529628997933   *
    DEC       -45.0                   -> -10.118127176345915   *
```

The second line checks the marker against the OBSID of the headers actually loaded, and prints
`<- MISMATCH` when they differ — which is exactly what `DONUT_SERVER_REPLAY_OBSID` pinning the wrong
exposure looks like. `EMUIMAGE` gets its own line rather than a row in the table because the
*original* has no `EMUIMAGE` (only replays do), so a live-vs-substituted column would show every
correct substitution as a change to `<absent>`.

## Diagnostic plots

Every finished job's three diagnostic figures are drawn automatically and shown on
`/results/<job_id>`, the page the dashboard's job table links to. Thumbnails inline, full-res on
click.

That page steps between exposures with the **← exposure / exposure →** buttons or the **left and
right arrow keys** (ignored while a text field has focus, where the arrows are the caret's). Stepping
switches the job in place rather than reloading, so the column picker, the sort, the row filter and
rows/page all carry over — which is the point, since comparing the same few columns across exposures
is what the arrows are for. `GET /admin/exposures` is what they walk: it lists only jobs that still
hold a result table, newest first, so an arrow can never land on a page that 409s.

They are **not** drawn by the coordinator. Plotting measured ~16 s a job on the r_57 exposure (0.6 s
reading the tables, 5 s building the figures, 10 s writing the six PNGs) against a ~30 s cadence, and the
deferred hook that writes the stamps runs inside the coordinator's serial command loop — which does
not return to `recv()` until it finishes, so drawing there would delay the *next* visit's `/prepare`
by 17 s. Instead a separate pool of **plot flights** draws them:

```zsh
bin/donutServer.py --num-plot-flights 2    # default 1; 0 disables plotting entirely
```

The queue between the two is **LIFO and uncapped**, which is a deliberate pair of choices:

- **Newest first**, because when a flight frees up the most recently finished job is the one someone
  is most likely looking at.
- **Uncapped**, because observing has gaps in which a backlog drains, and a cap would mean choosing
  at enqueue time which job never becomes viewable. The cost of a queued entry is a job id — the
  ~38 MB it needs is already on disk.

The consequence is accepted rather than prevented: at one flight (~16 s a job) against two Zernike
flights in steady state (~15 s a job), **the queue grows during continuous observing and its oldest
entries may never be drawn.** `plot_queue.depth` is on `/health` and on the dashboard so saturation
is visible rather than silent; `--num-plot-flights` is the dial.

The queue lives in the front-end's memory and dies with it, exactly as `JOBS` does. Two ways back:

- `POST /admin/plots/<job_id>` — one job, also the retry path for a failed draw. The results page's
  "draw plots" button.
- `POST /admin/plots` — scan `DONUT_SERVER_STAMP_DIR` for jobs with a result table but no plots and
  enqueue them all, newest first. The dashboard's "backfill missing plots" button.
- `GET /admin/plots` — the same scan without the enqueue, which is what that button's
  "*N* on disk · *M* missing plots" note polls. Needs no plot flight, so it still answers under
  `--num-plot-flights 0`.

A plot flight is a `Coord` like a compute flight, so it restarts and degrades the same way — and
`degraded` is terminal until a restart, which for these needs **`POST /admin/restart?pool=plot`**
(the bare route still means the compute flights, as it always did). The dashboard offers the button
on a degraded plot row. A job whose flight has no usable child is retried a few times and then
recorded as an error rather than left queued forever, so the results page always settles on
something an operator can act on.

### What it costs on disk

Output lands in `$DONUT_SERVER_STAMP_DIR/plots/<job_id>/`, six PNGs a job (three figures × full and
a ~1200 px thumbnail), and drawing needs two more parquet files written by the same deferred step as
the stamps. **All of it is unpruned, like the stamps.** Measured on the r_57 exposure:

| file | size |
|---|---|
| `<job_id>.parquet` (the stamps, pre-existing) | 18.4 MB |
| `<job_id>.detector_images.parquet` | **34.7 MB** |
| `<job_id>.selection_overlays.parquet` | 0.03–0.15 MB |
| `plots/<job_id>/*.png` (six files) | 21.3 MB |

That is ~75 MB a job against ~18 MB before, so **the disk rate roughly quadruples, to ~8.9 GB/hour**
at a 30 s cadence. The binned post-ISR images in `detector_images` are the bulk of it, and they
exist only for the focal-plane plot. Two ways to spend less:

- `-c doSelectionOutput=False` — no `detector_images`, no focal-plane plot, two figures instead of
  three. Takes a job to ~29 MB.
- `--num-plot-flights 0` — no plots, no PNGs, and no extra tables either: the coordinator reads the
  same variable and skips writing what nothing will read. Back to ~18 MB, the pre-plotting rate.

Everything here is safe to delete at any time; losing a plot costs only that plot, and
`POST /admin/plots` redraws it from the stamp table — provided *that* has not been deleted too.

Two endpoints serve the results page, both answering **409 while a plot does not exist yet** — the
ordinary state for the ~17 s after a job reports DONE, exactly as `/result/<job_id>/images` does:

- `GET /result/<job_id>/plots` — the manifest: which figures exist, their sizes, and the queue state
  (`queued`, `drawing`, `ready`, `error`, `absent`).
- `GET /result/<job_id>/plot/<name>?size=full|thumb` — the PNG. `name` is one of `donut`,
  `wavefront`, `selection`.

## Verification

```zsh
python -m pytest tests/ -q                      # 294 tests; the butler-backend 14
                                                # skip unless DONUT_SERVER_BUTLER_REPO is set,
                                                # and the RUN-fallback one also needs a repo
                                                # whose chain includes the producer runs
scons                                           # same suite, but sconsUtils rebuilds the
                                                # environment from an allowlist -- a
                                                # DONUT_SERVER_* variable missing from
                                                # tests/SConscript does not reach pytest
python -m lsst.ts.donut_server.coordinator      # full prepare -> push, no FastAPI
bin/donutServer.py                              # then, in another shell:
                                                # (-b <repo> for butler-resolved calibs)
bin/donutClient.py --visit 2026071300478 --wait 60
bin/donutClient.py --visit 2026071300478 --wait 60 --images   # also saves the stamps
bin/donutClient.py --visit 2026071300478 --wait 60 -c maxFitScatter=2.0
curl -si localhost:8000/config | head -20       # 200, X-Donut-Stale: 0
```

### Rehearsing a replay

Substitution now happens on the server, and `tests/test_replay.py` covers which exposure it picks —
but nothing automated runs a substituted header all the way through assembly to a Zernike, so that
is still checked by hand, which is also how to rehearse before BTS access. `scripts/replay_drill.py` builds a fake replayed
exposure from a real one, in two steps: it prints the expected answer and the trigger command for a
second terminal, then waits on Enter and copies all 8 sensors in.

```zsh
export DONUT_SERVER_REPLAY_DIR=$PWD/replay
scripts/replay_drill.py                                  # first exposure in the tree
scripts/replay_drill.py --source replay/MC_O_20260713_000250
scripts/replay_drill.py --emuimage MC_O_99999999_000001  # deliberate mismatch
scripts/replay_drill.py --no-pause                       # unattended
```

The copy is one shot rather than drip-fed: whether the trigger counts sensors up to 8 and waits for
their sizes to settle is the polling logic's own business, covered directly in
`tests/test_trigger.py`. What the drill is for is the part no test can reach — that substituted
headers survive assembly and come back as the *original* exposure's Zernikes.

The mangling is the point: the copy keeps the pixels but gets a different filter, a far-away
pointing, another night's timestamps and `FOCUSZ 0`, so a run that *silently failed to substitute* is
distinguishable from one that worked. Stage 1 prints the expected answer before anything runs —
compare against that rather than deciding afterwards whether the output looked about right.

**Do not judge it by `DONE`.** Both the substituted and unsubstituted runs succeed; see
[Replayed exposures at BTS](#replayed-exposures-at-bts) for why the service cannot tell them apart.
On `MC_O_20260713_000050`'s pixels the substituted run is identifiable by
`visit=2026071300050 filter=r_57 boresight=(215.5361, -10.1181)` and 8 of 8 groups fit, whatever the
files are labelled.

Four paths worth walking at least once, all verified:

| run | expected |
|---|---|
| `DONUT_SERVER_REPLAY_DIR` set | the original's `r_57` and pointing, `EMUIMAGE ... -> loaded ...` agreeing |
| unset | the mangled `g_6` and `(120, -45)`, submitted as-is — the deliberate wrong-sky test |
| `--emuimage` naming an absent exposure | hard failure naming the directory and suggesting `DONUT_SERVER_REPLAY_OBSID` |
| a real exposure (`replay/MC_O_*`) with the variable set | hard failure listing all 8 detectors as lacking `EMUIMAGE` |

**Check the substituted run, not just that the unsubstituted one differs.** Both come back `DONE` —
see [Replayed exposures at BTS](#replayed-exposures-at-bts) for why the service cannot tell the
difference. The substituted one is identifiable by `physical_filter: r_57` in the quantum and 8 of 8
groups fit; the unsubstituted one reports `z_20` and fits 6 of 6 of a different, meaningless grouping.

The dashboard is at <http://127.0.0.1:8000/dashboard> and needs no credential: the server exempts
loopback callers and serves the page to nobody else, so reaching it from another machine means an
`ssh -L 8000:localhost:8000` tunnel. `DONUT_SERVER_TOKEN` is still what protects `/prepare` and
`/push` from the network, so a producer on a *different* host does need `--token`.

It lays the flights out side by side, with one job state machine per flight, since with
`--num-flights 2` two jobs are in flight at once. `rss` on each flight is that coordinator's own
resident size (4–6 GB in steady state), *not* its process group's: the task's eight cutout workers
share those pages copy-on-write, so summing the group would report tens of GB that do not exist. The
header carries the front-end's own pid — the one to signal — its RSS, and the total. A flight reading
`degraded` has given up relaunching its coordinator: it holds no child and serves no job until
`POST /admin/restart?flight=N`, either because bring-up failed `MAX_RESTART_ATTEMPTS` times running
or because the child reported a config error retrying cannot fix. The other flights keep serving,
which is the `partial` in `/health`.

Each flight also reports `exposures`: completed pushes on that lane since the **front-end** came up,
with the pool's total in the section header. It deliberately survives a coordinator restart — the
question it answers is whether a lane is pulling its weight, and a counter that zeroed on every
respawn would look like that one while answering something else. A plot flight's row reports the same
field as `drawn`, counting completed draws.

The job table's rows are clickable: a click **pins** that job, and the detail panel below — phase
bar, prepare timings, summary, and the quantum (`visit`, `group`, `n_raws`, `n_refcat_shards`,
`n_input_datasets`) — follows the pinned job instead of the newest one. Click the same row again,
press Escape, or use the *unpin* button to go back to following. The pin never touches the per-flight
state machines, which always show live state, and it releases itself if the pinned job falls off the
end of `limit`.

Below the per-flight job panels, **plot flights** get one compact row each — no state machine, since
nothing is blocked on a plot and the only questions are whether it is up and how far behind. Each row
carries the same `ready` and `busy`/`idle` chips the compute flights use, plus which job it is drawing
(linked to that job's results page). The section header carries the queue depth; a `queued` count that
only grows means plotting cannot keep up at this cadence, so raise `--num-plot-flights`.

The backfill button's note is live, not frozen at the last click: it re-scans every 5 s — slower than
the 1 Hz the rest of the page polls at, because it walks the stamp directory — and immediately
whenever a job settles or a draw finishes, the two events that move the numbers.

An open dashboard is about 2.5 requests a second, so its **successful** polls (`/health`,
`/admin/jobs`, `/admin/log`, `/admin/plots`, `/admin/exposures`) are dropped from uvicorn's access
log — otherwise they bury the one line a push emits and the terminal is useless for watching compute.
Only those paths, only GET, and only 2xx: a 503 from `/health`, a 401, a POST to `/admin/plots` and
everything on the compute path are all still logged. `--access-log-polls` turns the filter off.

Acceptance criteria for the r_57 exposure — a packaging or refactoring change touches no compute
path, so any movement here means import order or thread clamping regressed:

- boresight `ra=283.6660 dec=-28.1326`
- prepare: `n_level5_files: 10`, `n_level7_shards: 160`, `n_rows: 1148693`
  (~0.5–0.7 s cold, ~0.0003 s on a repeat)
- prepare `task`: `reused: false` and ~0.14 s on the first prepare or any changed override list,
  `reused: true` and ~0 s on a repeat. `calib` must stay `reused: true` across an override-only
  change — if it reloads, the two guards have been folded together
- prepare `calib`: a *filter* change must report `detector_reused: true` and `filter_reused: false`.
  If both go false, the two calib halves have been folded back together. Holds under either backend —
  the detector half's datasets do not depend on the filter, so the butler resolves it to the same ids
- `n_input_datasets: 208`
- **63 donut rows** across 8 detectors, 28/29 groups fit
- `donut_id` values are Gaia source ids (e.g. `6761235373898405888`) — the quickest confirmation the
  refcat path ran rather than a blind fallback, as is the *absence* of the "No reference catalog
  shards provided" warning
- **Measure steady state, not the first job:** the first `runQuantum` in a fresh coordinator costs
  7.5–8.9 s; jobs 2 onward settle at 6.1–6.7 s
- `coordinator_total_s` must not include the image write. The stamps cost ~0.1 s, spent *after* the
  reply, so they appear in the log as `wrote .../<job_id>.parquet ... in 0.100s` and nowhere in the
  push timings — on a clean run the phases sum to within ~0.02 s of `coordinator_total_s`, leaving no
  room for it. If that time shows up in `push_timings`, the write has drifted onto the critical path.
