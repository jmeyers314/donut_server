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

## Data

Not in the repo (33 GB, git-ignored). Locate it yourself — each directory is found through its own
environment variable, so the data need not live in the checkout:

| variable | size | contents |
|---|---|---|
| `DONUT_SERVER_RAW_DIR` | 5.1 G | real corner-sensor raws, overscan present, ISR-ready |
| `DONUT_SERVER_CALIB_DIR` | 4.8 G | ptc/linearizer/crosstalk per detector; flats and intrinsic Zernikes per band |
| `DONUT_SERVER_REFCAT_DIR` | 23 G | Gaia level-5 shards, resharded to level 7 at prepare time |

`DONUT_SERVER_STAMP_DIR` is the one output directory, and it is required too. It needs no data up
front, only somewhere writable: the coordinator puts one `<job_id>.parquet` there per job, holding
the full result table including the per-donut stamps.

```zsh
export DONUT_SERVER_CALIB_DIR=$PWD/calib
export DONUT_SERVER_REFCAT_DIR=$PWD/ref_cat
export DONUT_SERVER_RAW_DIR=$PWD/raw
export DONUT_SERVER_STAMP_DIR=$PWD/stamps
```

Unset is a loud `RuntimeError` naming the variable, not a silent empty result. The data-dependent
tests skip when unset, so `scons`/`pytest` still pass without the data — check the skip count.

**These files are never pruned, and at a 30 s cadence they arrive at ~2.0 GB/hour** (16.7 MB a job,
measured on the r-band exposure below — the in-memory columns are ~21 MB and parquet compresses).
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

## Verification

```zsh
python -m pytest tests/ -q                      # 93 tests
python -m lsst.ts.donut_server.coordinator      # full prepare -> push, no FastAPI
bin/donutServer.py                              # then, in another shell:
bin/donutClient.py --visit 2026071300478 --wait 60
bin/donutClient.py --visit 2026071300478 --wait 60 --images   # also saves the stamps
```

The dashboard is at <http://127.0.0.1:8000/dashboard> and needs no credential: the server exempts
loopback callers and serves the page to nobody else, so reaching it from another machine means an
`ssh -L 8000:localhost:8000` tunnel. `DONUT_SERVER_TOKEN` is still what protects `/prepare` and
`/push` from the network, so a producer on a *different* host does need `--token`.

Acceptance criteria for the r-band exposure — a packaging or refactoring change touches no compute
path, so any movement here means import order or thread clamping regressed:

- boresight `ra=283.6660 dec=-28.1326`
- prepare: `n_level5_files: 10`, `n_level7_shards: 160`, `n_rows: 1148693`
  (~0.5–0.7 s cold, ~0.0003 s on a repeat)
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
