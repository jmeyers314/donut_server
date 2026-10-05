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
it). The server never opens a raw — they arrive over shared memory on `/push` —
so a server host needs neither the variable nor the 5.1 G.

`DONUT_SERVER_STAMP_DIR` is the one output directory, and it is required too. It needs no data up
front, only somewhere writable: the coordinator puts one `<job_id>.parquet` there per job, holding
the full result table including the per-donut stamps.

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
measured on the r_57 exposure below — the in-memory columns are ~21 MB and parquet compresses).
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
- **`-c savePlots=True` writes PNGs into the server's working directory** (two per job, unpruned,
  named only for the visit) and puts plot time on the push critical path. `DonutBlitzPlotTask` is
  designed to be run later from the retained in-memory table instead; prefer that.
- **An override can be rejected.** Anything that perturbs the task's *connection set* (notably
  `-c doZernikesOutput=True`, which adds a second output this service builds no ref for) is a **400
  at prepare**, not a mysterious `KeyError` mid-push. Likewise a positive `hangTimeout` at or below
  `unitTimeout`, which would make the watchdog abort the coordinator on every ordinary job. In both
  cases the previously good config keeps serving.

## Verification

```zsh
python -m pytest tests/ -q                      # 185 tests; the butler-backend 14
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

The dashboard is at <http://127.0.0.1:8000/dashboard> and needs no credential: the server exempts
loopback callers and serves the page to nobody else, so reaching it from another machine means an
`ssh -L 8000:localhost:8000` tunnel. `DONUT_SERVER_TOKEN` is still what protects `/prepare` and
`/push` from the network, so a producer on a *different* host does need `--token`.

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
