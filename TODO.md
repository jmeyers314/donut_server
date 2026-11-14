# TODO

Scratch list, arbitrary order.

- [ ] **Eliminate `donutTrigger.py`** — run production through the camera's
      per-file script instead. `scripts/camera_hook.sh` is the deployed shape,
      and it does the whole job with a single `curl` to `/submit`, no Python
      startup and no polling (the hook's own `flock` already guarantees what the
      polling was for).

      What `donutTrigger.py` still uniquely offers, and so has to be accounted
      for before it goes: `--dry-run` against a new deployment, `--wait`, and the
      poll itself for any caller that is *not* the camera hook.

- [ ] **Make the calib backend exclusively butler** — drop `raw/` and `calib/`
      and the `files` backend with them: `DONUT_SERVER_CALIB_DIR`,
      `DONUT_SERVER_RAW_DIR`, and the flat-tree loader. Now unblocked by having
      scripts and procedures for extracting just the parts of a repo we need.

      Removes the two-backends-must-agree burden, and with it the flat tree's
      inability to answer the question production actually asks: *which* calib
      was valid when this exposure was taken.

- [ ] **Try other HTM shard levels for the refcat** — if HTM7 is not worse than
      HTM5, we can use it more directly and skip the reshard-at-prepare step.
      **More thoughts to come on this one** — recorded here only so it is not
      lost.

- [ ] **Get ref cat frome the butler too**

- [ ] **Numpy docstrings throughout** — one file at a time, deliberately, so
      each pass doubles as a review of the file it touches. `protocol.py` is
      done (`0b4892f`).

- [ ] **Pre-commit checks** — configured but deliberately *not* installed as a
      git hook; run by hand per file as each is brought into compliance:

          pre-commit run --files python/lsst/ts/donut_server/protocol.py

- [ ] **License headers** — `protocol.py` has its 21 lines of GPLv3; the other
      sources don't. `insert-license` will add them in one pass when we're
      ready.

- [ ] **Simplify the docs** — `README.md` and `Rubin Wavefront Estimation
      Service.md` are both far too complicated right now. Probably a lot of the
      code comments too: several carry measurement logs and rationale that have
      outlived the decision they justified.

- [ ] **Audit for other simplifications** — open-ended sweep, in the spirit of
      the item above.

- [ ] **Think through replay/emuimage header handling trigger** - might be
      better to trigger on header presence instead of env var presence.

- [ ] **Investigate Server Side Events (SSE) for dashboard.**

- [ ] **Rethink where the config should live.**  Also nproc.  Maybe default in the
      dashboard followed by http overrides?
