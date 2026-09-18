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
setup -kr ~/src/ts_wep      # a working copy; ts_wep is not eups-installed
```

`ts_wep` is set up from a checkout deliberately: the service tracks unreleased blitz-task changes.

### `.env` is not in the repo

A dumped-environment `.env` is git-ignored, so a fresh clone has none and `run_server.sh` /
`run_client.sh` will fail on `source .env` until you regenerate one from an eups-setup shell:

```zsh
env > .env
```

It is a shortcut for skipping the `setup` calls above, not a supported entry point. Note that
`/usr/bin/env` is SIP-protected on macOS and strips `DYLD_LIBRARY_PATH` from child processes, which
is why the launcher scripts remap `LD_LIBRARY_PATH` onto it — without that, `import lsst.afw.image`
fails with `Library not loaded: libbase.dylib`.

## Data

Not in the repo (33 GB, git-ignored), found relative to the source tree:

| dir | size | contents |
|---|---|---|
| `raw/` | 5.1 G | real corner-sensor raws, overscan present, ISR-ready |
| `calib/` | 4.8 G | ptc/linearizer/crosstalk per detector; flats and intrinsic Zernikes per band |
| `ref_cat/` | 23 G | Gaia level-5 shards, resharded to level 7 at prepare time |

## Verification

```zsh
python -m pytest tests/ -q          # 53 tests
python coordinator.py               # full prepare -> push, no FastAPI
./run_server.sh                     # then, in another shell:
./run_client.sh --token <tok> --visit 2026071300478 --wait 60
```

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
