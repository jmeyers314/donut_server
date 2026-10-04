#!/usr/bin/env python
"""Start the donut_server FastAPI app.

Environment setup is the caller's business: source loadLSST.zsh, `setup
lsst_distrib`, `setup -kr ~/src/ts_wep`, `setup -kr .`, and point the three
DONUT_SERVER_*_DIR variables at your data. See README.md.

There is no DYLD_LIBRARY_PATH remap here, unlike the run_server.sh this
replaces. macOS SIP strips DYLD_* from children of system-volume binaries, and
`#!/usr/bin/env python` was one -- but scons rewrites that shebang to the
absolute conda interpreter, which is not SIP-protected, so DYLD_LIBRARY_PATH
survives into this process and the remap is dead code. Running this file
straight out of bin.src/ rather than the built bin/ will fail to import afw for
exactly that reason.
"""
import argparse
import os
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    # Both default to None and are written into the environment below, because that
    # is the only transport that reaches where they are read: uvicorn.run() imports
    # the app by string in this process, and the coordinator children inherit the
    # environment from it. The flags exist for discoverability through --help.
    # Short forms follow `pipetask run`: -j is the process count
    # (--processes there), -n the cores one quantum may use
    # (--cores-per-quantum).
    parser.add_argument(
        "-j",
        "--num-flights",
        type=int,
        help="coordinator processes to run, each able to compute one job at a time "
        "(default 2; DONUT_SERVER_NUM_FLIGHTS)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        help="cores the task may fork over per push, not divided between flights "
        "(default 8; DONUT_SERVER_NUM_WORKERS)",
    )
    # Giving a repo is what selects the butler calib backend -- there is no
    # --calib-backend flag, because the repo is the only thing that backend needs
    # that the files one cannot use. Short forms match donutClient.py's -b/-i,
    # which in turn match `pipetask run`.
    parser.add_argument(
        "-b",
        "--butler-repo",
        help="butler repo root or alias to resolve calibs from, valid for the "
        "exposure's time; without this they are read from the flat FITS tree "
        "under DONUT_SERVER_CALIB_DIR (DONUT_SERVER_BUTLER_REPO)",
    )
    parser.add_argument(
        "-i",
        "--butler-collections",
        help="comma-separated collections to resolve calibs from, highest priority "
        "first; only meaningful with --butler-repo "
        "(default LSSTCam/defaults; DONUT_SERVER_BUTLER_COLLECTIONS)",
    )
    args = parser.parse_args()

    # These flags *are* a way of setting the environment variables, which is the
    # only transport that reaches where they are read -- so a flag assigns,
    # overriding whatever was exported. Each defaults to None, and None means
    # "not given", leaving an exported value in place. Precedence is therefore
    # flag > exported > default, the usual way round.
    for flag, var in (
        (args.num_flights, "DONUT_SERVER_NUM_FLIGHTS"),
        (args.num_workers, "DONUT_SERVER_NUM_WORKERS"),
        (args.butler_repo, "DONUT_SERVER_BUTLER_REPO"),
        (args.butler_collections, "DONUT_SERVER_BUTLER_COLLECTIONS"),
    ):
        if flag is not None:
            os.environ[var] = str(flag)

    # A repo implies the butler backend; no repo implies the files one. The
    # coordinator still reads DONUT_SERVER_CALIB_BACKEND -- it is the backend's
    # own contract and what the tests set -- so this derives it rather than
    # replacing it.
    #
    # Naming a repo on the command line is as explicit as it gets, so it wins
    # over an exported backend; a repo that arrived in the environment only
    # defaults it, leaving an exported DONUT_SERVER_CALIB_BACKEND=files able to
    # pin the files backend while a repo stays exported for another run.
    if args.butler_repo is not None:
        os.environ["DONUT_SERVER_CALIB_BACKEND"] = "butler"
    elif os.environ.get("DONUT_SERVER_BUTLER_REPO"):
        os.environ.setdefault("DONUT_SERVER_CALIB_BACKEND", "butler")

    # Collections without a repo would otherwise be silently ignored: they only
    # ever reach the butler backend, so this is someone expecting calibs to come
    # from a repo they have not named.
    if os.environ.get("DONUT_SERVER_BUTLER_COLLECTIONS") and not os.environ.get(
        "DONUT_SERVER_BUTLER_REPO"
    ):
        parser.error(
            f"{'--butler-collections' if args.butler_collections else 'DONUT_SERVER_BUTLER_COLLECTIONS'}"
            " needs a repo; pass --butler-repo (or export DONUT_SERVER_BUTLER_REPO). "
            "Without one, calibs come from DONUT_SERVER_CALIB_DIR and no "
            "collection is consulted"
        )

    # An exported backend with no repo to resolve from. Caught here because
    # otherwise it comes up looking healthy: nothing on the startup path resolves
    # a calib, so both flights spawn, /health reports ready, and every /prepare
    # then fails with a 500.
    #
    # That the repo *resolves* is deliberately not checked: that needs
    # Butler.from_config, and keeping the stack out of this process is
    # load-bearing (uvicorn.run imports the app in-process below). The
    # coordinator validates it at the first resolve.
    if os.environ.get("DONUT_SERVER_CALIB_BACKEND") == "butler" and not os.environ.get(
        "DONUT_SERVER_BUTLER_REPO"
    ):
        parser.error(
            "DONUT_SERVER_CALIB_BACKEND=butler needs a repo; pass --butler-repo "
            "(or export DONUT_SERVER_BUTLER_REPO)"
        )

    # Before importing anything that pulls numpy: keep BLAS/OpenMP
    # single-threaded so the task's 8 fork workers don't each fork with a live
    # thread pool. This is for fork safety, not speed -- 1 vs 12 threads is a
    # wall-time dead heat, but this numpy links OpenBLAS built with USE_OPENMP,
    # and forking with live OpenMP threads is undefined behaviour.
    #
    # Setting these in-process is sufficient, and measured: neither uvicorn nor
    # fastapi imports numpy, so nothing has bound a thread pool by the time
    # coordinator is imported. coordinator._assert_single_threaded_blas()
    # verifies it at startup regardless.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    # Owned here so the path can be announced; logtail falls back to the same
    # cwd default when this is unset (e.g. coordinator's own smoke test).
    log = os.environ.setdefault(
        "DONUT_SERVER_LOG", os.path.join(os.getcwd(), "donut_server.log")
    )
    print(f"Pipeline logs (INFO and above) -> {log}")
    print(f"  tail -f {log}")

    # Announced because "how many lanes is this running" is the first thing to
    # check when throughput looks wrong.
    print(
        f"Flights: {os.environ.get('DONUT_SERVER_NUM_FLIGHTS', '2')}"
        f"  workers/push: {os.environ.get('DONUT_SERVER_NUM_WORKERS', '8')}"
    )

    # Which calibs are in use, for the same reason -- and since the backend is
    # now inferred from whether a repo was given, the line has to name the source
    # it chose, not just the choice. Read back out of the environment so it
    # reports what the coordinator will actually see, including values that came
    # from the shell rather than a flag.
    backend = os.environ.get("DONUT_SERVER_CALIB_BACKEND", "files")
    if backend == "butler":
        collections = os.environ.get("DONUT_SERVER_BUTLER_COLLECTIONS", "LSSTCam/defaults")
        print(f"Calibs: butler  repo={os.environ.get('DONUT_SERVER_BUTLER_REPO', '<unset>')}")
        print(f"  collections={collections}")
    elif backend == "files":
        # Not resolved through coordinator.calib_dir(), which raises when unset:
        # importing it would pull the stack into this process. An unset dir is
        # the coordinator's error to report, so just say it is unset.
        print(
            f"Calibs: files  dir="
            f"{os.environ.get('DONUT_SERVER_CALIB_DIR', '<unset>')}"
        )
    else:
        # Reported rather than folded into the files branch, which would print a
        # calib dir this server is never going to read. Not rejected here: the
        # valid set belongs to coordinator.calib_backend(), which raises naming
        # it -- but only at the first /prepare, so say so now.
        print(f"Calibs: {backend!r} is not a known backend; /prepare will fail")

    # Still generated even though the dashboard no longer asks for one: the token is
    # what protects /prepare and /push from the network. Callers on this host are
    # exempt, so it is only needed by a producer running somewhere else.
    if not os.environ.get("DONUT_SERVER_TOKEN"):
        token = secrets.token_hex(16)
        os.environ["DONUT_SERVER_TOKEN"] = token
        print("DONUT_SERVER_TOKEN not set; generated one for this run:")
        print(f"  {token}")
        print("Only remote clients need it; on this host nothing does:")
        print(f"  donutClient.py --host http://{args.host}:{args.port} --token {token} --visit <visit>")

    import uvicorn

    # By import string rather than by object: uvicorn resolves it after its own
    # setup, and this is the form that supports --reload if it is ever wanted.
    uvicorn.run("lsst.ts.donut_server.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
