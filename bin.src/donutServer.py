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
    parser.add_argument(
        "--num-flights",
        type=int,
        help="coordinator processes to run, each able to compute one job at a time "
        "(default 2; DONUT_SERVER_NUM_FLIGHTS)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="cores the task may fork over per push, not divided between flights "
        "(default 8; DONUT_SERVER_NUM_WORKERS)",
    )
    args = parser.parse_args()

    # setdefault, so an already-exported env var wins over the flag's default of
    # None -- and an explicit flag wins over nothing, which is the intent.
    for flag, var in (
        (args.num_flights, "DONUT_SERVER_NUM_FLIGHTS"),
        (args.num_workers, "DONUT_SERVER_NUM_WORKERS"),
    ):
        if flag is not None:
            os.environ.setdefault(var, str(flag))

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
