# TODO

Scratch list, arbitrary order.

- [ ] **Rename `/ingest`** — it suggests `butler ingest-raws`, registering files
      into a repo, which is not what it does: it touches no registry and writes
      nothing. The in-memory butler it appears to name is built in `run_job`,
      which `/push` reaches too, so the name does not even distinguish the two
      paths. What it actually means is pixels by reference rather than by value,
      so `/pull` would be both accurate and symmetric with `/push`.

      Not just the route: `cmd: "ingest"` on the coordinator wire, `run_ingest`,
      and the `ingest_s` timing key, which the dashboard charts and job records
      may hold.

- [ ] **A producer CLI for `/submit`** — deleting `donutTrigger.py` left no
      ergonomic way to submit one directory by hand; it is a `curl` with up to
      six optional JSON fields. Belongs in `donutClient.py` as a mode, which
      already has the token handling and the `-c`/`-C` override parsing.

      Also gone with it: the poll. Nothing needs it today, since the camera's
      hook counts sensors under its own `flock`, but pointing a producer at a
      directory that is still filling is the BTS scenario to remember.

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

- [ ] **`pathlib` instead of `os.path`** — repo-wide, in one pass rather than
      file by file, since half-converted path handling is worse than either.

- [ ] **Simplify the docs** — `README.md` and `Rubin Wavefront Estimation
      Service.md` are both far too complicated right now. Probably a lot of the
      code comments too: several carry measurement logs and rationale that have
      outlived the decision they justified.

- [ ] **Audit for other simplifications** — open-ended sweep, in the spirit of
      the item above.

- [ ] **Investigate Server Side Events (SSE) for dashboard.**

- [ ] **Rethink where the config should live.**  Also nproc.  Maybe default in the
      dashboard followed by http overrides?
