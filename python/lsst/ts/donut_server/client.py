"""Test producer for the mock WF estimation service. Single-shot (--once,
default) or cadence-loop (--loop --interval 30) mode.

Sends the real corner-sensor raws for one exposure, sourced either from a
directory of `raw_*.fits` files (default) or from a butler repo (--butler),
which is how you drive the service from a host that has many raws.
"""
from __future__ import annotations

import argparse
import glob
import ipaddress
import os
import re
import socket
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from lsst.ts.donut_server import exposure_codec
from lsst.ts.donut_server import protocol

def raw_dir() -> str:
    """The raw exposure directory, located by the caller's environment.

    Only the default for --raw-dir, so it is resolved when the CLI is built
    rather than at import.
    """
    d = os.environ.get("DONUT_SERVER_RAW_DIR")
    if not d:
        raise RuntimeError(
            "DONUT_SERVER_RAW_DIR is not set; point it at the directory holding "
            "raw_*.fits, or pass --raw-dir (see README)."
        )
    return d

def host_is_loopback(host: str) -> bool:
    """Whether --host points at this machine, which needs no token.

    Resolves a name rather than only matching literals, because "localhost" is at
    least as likely to be typed as "127.0.0.1". Anything unresolvable counts as
    remote: the wrong answer there is only a token requirement, whereas the wrong
    answer the other way is a confusing 401 from the server.
    """
    name = urllib.parse.urlsplit(host).hostname
    if not name:
        return False
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(name, None)
    except socket.gaierror:
        return False
    return bool(infos) and all(
        ipaddress.ip_address(info[4][0]).is_loopback for info in infos
    )


_RAW_RE = re.compile(r"raw_(\d+)_(\d+)_([a-z]+)\.fits")

# physical_filter -> band for the six survey filters this service has calibs for.
# Read off the raws in `raw/` themselves, whose FilterLabel carries both labels;
# it is the same set `calib/` holds flat_<det>_<band>.fits for. Anything else is
# a hard error rather than a guess, because the band is what picks the flat, and
# a filter we have not seen is one whose calibs nobody has checked.
BAND_BY_PHYSICAL_FILTER = {
    "u_24": "u",
    "g_6": "g",
    "r_57": "r",
    "i_39": "i",
    "z_20": "z",
    "y_10": "y",
}


@dataclass(frozen=True)
class RawSource:
    """One exposure's corner-sensor raws, located but not yet read.

    Handles rather than loaded `ExposureF`s so the send path keeps its two
    phases: /prepare goes out knowing only the pointing, and the pixels are
    read afterwards. That mirrors a real producer, where the boresight is known
    at shutter open and pre-loading refcat shards for it is off the push
    critical path.
    """

    visit: int
    band: str
    handles: dict[int, Any]
    load: Callable[[Any], Any]
    # Set when the pointing is available without reading pixels (the butler
    # registry knows it); None means read it off the first raw's VisitInfo.
    boresight: tuple[float, float] | None = None


def discover_exposures(raw_dir: str) -> dict[int, tuple[str, dict[int, str]]]:
    """Scan `raw_dir` for raw_<visit>_<detector>_<band>.fits.

    Returns {visit: (band, {detector_id: path})}.
    """
    exposures: dict[int, tuple[str, dict[int, str]]] = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "raw_*.fits"))):
        m = _RAW_RE.fullmatch(os.path.basename(path))
        if not m:
            continue
        visit, det_id, band = int(m.group(1)), int(m.group(2)), m.group(3)
        entry = exposures.setdefault(visit, (band, {}))
        if entry[0] != band:
            raise RuntimeError(f"visit {visit} has mixed bands: {entry[0]!r} and {band!r}")
        entry[1][det_id] = path
    return exposures


def resolve_exposure(raw_dir: str, visit: int) -> tuple[int, str, dict[int, str]]:
    """Locate one visit's raws. The band comes back as a property of the visit,
    read off its filenames, so it cannot disagree with the pixels being sent."""
    exposures = discover_exposures(raw_dir)
    if not exposures:
        raise RuntimeError(f"no raw_*.fits files found in {raw_dir!r}")
    if visit not in exposures:
        raise RuntimeError(
            f"visit {visit} not found in {raw_dir!r}; have {sorted(exposures)}"
        )

    band, paths = exposures[visit]
    return visit, band, paths


def resolve_from_files(raw_dir: str, visit: int) -> RawSource:
    """A RawSource backed by `raw_<visit>_<detector>_<band>.fits` files."""
    import lsst.afw.image as afwImage

    chosen_visit, band, paths = resolve_exposure(raw_dir, visit)
    return RawSource(
        visit=chosen_visit,
        band=band,
        handles=dict(paths),
        load=afwImage.ExposureF.readFits,
    )


def resolve_from_butler(
    butler: Any, instrument: str, collections: list[str], visit: int
) -> RawSource:
    """A RawSource backed by `raw` datasets in a butler repo.

    Notes on the shape of this:

    - `handles` maps detector id -> `DatasetRef`, and `load` is `butler.get`.
      Only the 8 corner wavefront sensors belong in it: the server's calibs are
      keyed by detector name and it rejects a push naming any detector it has
      no calibs for.
    - The band is derived from the exposure's `physical_filter` via
      `BAND_BY_PHYSICAL_FILTER` and reported, never accepted from the caller.
      /prepare loads flats and intrinsic Zernikes for it, so a band that
      disagreed with the pixels would silently pair them with the wrong flat.
    - Raws are dimensioned by `exposure`, not `visit`; the integer is the same
      one the file naming calls a visit.
    - `boresight` comes off the exposure record's `tracking_ra` /
      `tracking_dec` (degrees), which keeps it off the pixel path entirely --
      the alternative is reading the first raw's VisitInfo at a full 114 MB get.

    Two traps in the butler query API, verified against a scratch repo with
    LSSTCam registered, worth remembering if these queries are ever reworked:

    - **Never name a bind parameter after a dimension.** `exposure = :exposure`
      parses as a tautology and returns *every* row with no error;
      `exposure = :e` filters correctly. A governor is worse-but-louder:
      `instrument = :instrument` raises InvalidQueryError.
    - `query_datasets` / `query_dimension_records` default to `explain=True`,
      which raises `EmptyQueryResultError` instead of returning `[]`.
    """
    (record,) = butler.registry.queryDimensionRecords(
        "exposure",
        where=f"exposure={visit} and instrument='{instrument}'"
    )

    refs = butler.query_datasets(
        "raw",
        collections=collections,
        where=(
            f"exposure={visit} "
            f"and instrument='{instrument}' "
            "and detector in (191,192,195,196,199,200,203,204)"
        ),
    )

    handles = {
        **{ref.dataId["detector"]: ref for ref in refs}
    }

    physical_filter = str(record.physical_filter)
    try:
        band = BAND_BY_PHYSICAL_FILTER[physical_filter]
    except KeyError:
        raise RuntimeError(
            f"visit {visit} has physical_filter {physical_filter!r}, which has no "
            f"known band; known: {sorted(BAND_BY_PHYSICAL_FILTER)}"
        ) from None

    return RawSource(
        visit=visit,
        band=band,
        handles=handles,
        load=butler.get,
        boresight=(record.tracking_ra, record.tracking_dec)
    )


def read_boresight(source: RawSource) -> tuple[float, float]:
    """Boresight (ra, dec) in degrees for `source`.

    Free if the source already knows the pointing; otherwise read off any one
    raw's VisitInfo.
    """
    if source.boresight is not None:
        return source.boresight

    exp = source.load(source.handles[min(source.handles)])
    boresight = exp.getInfo().getVisitInfo().boresightRaDec
    return boresight.getRa().asDegrees(), boresight.getDec().asDegrees()


def build_raw_parts(source: RawSource) -> dict[str, bytes]:
    """Read each raw and encode it into its two wire parts.

    Keyed by *detector name* (e.g. R00_SW0) taken off the exposure itself, so
    the coordinator can index _CALIB_STORE's *_by_name dicts directly.
    """
    parts: dict[str, bytes] = {}
    for det_id in sorted(source.handles):
        exp = source.load(source.handles[det_id])
        name = exp.getDetector().getName()
        img_blob, meta_blob = exposure_codec.encode_exposure(exp)
        parts[protocol.part_name(name, protocol.IMG_SUFFIX)] = img_blob
        parts[protocol.part_name(name, protocol.META_SUFFIX)] = meta_blob
    return parts


def run_once(
    host: str,
    token: str,
    calib_selector: str,
    source: RawSource,
    wait: float,
) -> None:
    # Omitted entirely rather than sent empty when there is no token: this client
    # needs none against a server on the same host, which exempts loopback.
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    print(
        f"exposure -> visit={source.visit} band={source.band} "
        f"detectors={sorted(source.handles)}"
    )

    boresight_ra, boresight_dec = read_boresight(source)
    print(f"boresight -> ra={boresight_ra:.4f} dec={boresight_dec:.4f}")

    t0 = time.monotonic()
    resp = requests.post(
        f"{host}/prepare",
        json={
            "band": source.band,
            "calib_selector": calib_selector,
            "boresight_ra": boresight_ra,
            "boresight_dec": boresight_dec,
        },
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    prepare_data = resp.json()
    job_id = prepare_data["job_id"]
    print(
        f"prepare -> job_id={job_id} timings={prepare_data.get('timings')} "
        f"({time.monotonic() - t0:.3f}s)"
    )

    t0 = time.monotonic()
    parts = build_raw_parts(source)
    blob = protocol.pack_blob(parts)
    print(
        f"encode -> {len(source.handles)} sensors, blob_bytes={len(blob)} "
        f"({time.monotonic() - t0:.3f}s)"
    )

    t0 = time.monotonic()
    resp = requests.post(
        f"{host}/push/{job_id}",
        data=blob,
        headers={**headers, "Content-Type": "application/octet-stream"},
        timeout=300,
    )
    resp.raise_for_status()
    push_data = resp.json()
    print(f"push -> state={push_data['state']} ({time.monotonic() - t0:.3f}s)")
    for key, value in (push_data.get("timings") or {}).items():
        print(f"    {key}: {value}")

    t0 = time.monotonic()
    resp = requests.get(
        f"{host}/result/{job_id}", params={"wait": wait}, headers=headers, timeout=wait + 30
    )
    resp.raise_for_status()
    result_data = resp.json()
    print(
        f"result -> ready={result_data['ready']} state={result_data['state']} "
        f"({time.monotonic() - t0:.3f}s)"
    )
    result = result_data.get("result") or {}
    if result.get("error"):
        print(f"  error: {result['error']}")
        return
    if result.get("quantum"):
        print(f"  quantum: {result['quantum']}")
    summary = dict(result.get("summary") or {})
    columns = summary.pop("columns", [])
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if not result_data.get("table_url"):
        return
    t0 = time.monotonic()
    resp = requests.get(f"{host}{result_data['table_url']}", headers=headers, timeout=60)
    resp.raise_for_status()
    print(f"  table -> {len(resp.content)} bytes parquet ({time.monotonic() - t0:.3f}s)")
    print_blitz_table(resp.content, columns)


def print_blitz_table(parquet_bytes: bytes, columns: list) -> None:
    """Deserialize the parquet body back into the astropy Table the task produced."""
    import io

    import pyarrow.parquet
    from lsst.daf.butler.formatters.parquet import arrow_to_astropy

    table = arrow_to_astropy(pyarrow.parquet.read_table(io.BytesIO(parquet_bytes)))
    print(f"  deserialized: {len(table)} rows x {len(table.colnames)} cols")
    assert list(table.colnames) == list(columns), "column mismatch vs server summary"

    table["det_name", "donut_id", "group_id", "group_fit_success", "snr"][:5].pprint()
    zk = table["zk_deviation_ccs"]
    print(f"  zk_deviation_ccs: shape={zk.shape} unit={zk.unit}")
    for i, row in enumerate(zk):
        print(f"  donut {i}, Noll 4-11: {[round(float(v), 4) for v in row[4:12]]}")


def report_transient_or_raise(exc: requests.HTTPError) -> None:
    """Print and swallow a 409/503; re-raise anything else."""
    response = exc.response
    status = response.status_code if response is not None else None
    if status not in (409, 503):
        raise exc

    detail = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        parts = [f"{key}={body[key]!r}" for key in ("reason", "error") if key in body]
        detail = " " + " ".join(parts) if parts else ""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        detail += f" retry_after={retry_after}s"
    print(f"transient {status}:{detail or ' no detail'} -- continuing to next cycle")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock producer for the WF estimation service")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.environ.get("DONUT_SERVER_TOKEN", ""))
    parser.add_argument("--calib-selector", default="default")
    # Defaults to None, not raw_dir(): resolving here would demand
    # DONUT_SERVER_RAW_DIR even from a --butler run that never reads files.
    parser.add_argument(
        "--raw-dir", default=None, help="directory of raw_*.fits files"
    )
    parser.add_argument(
        "--butler",
        default=None,
        help="butler repo (path or URI) to read raws from instead of --raw-dir",
    )
    parser.add_argument(
        "--collections",
        default="LSSTCam/defaults",
        help="comma-separated collections to search for raw; required with --butler",
    )
    parser.add_argument("--instrument", default="LSSTCam", help="instrument, with --butler")
    parser.add_argument(
        "--visit",
        type=int,
        required=True,
        help="visit to send. Its band is derived from the exposure itself, so "
        "there is no --band. Raws are exposure-dimensioned in the butler; this "
        "is that same integer.",
    )
    parser.add_argument("--wait", type=float, default=10.0, help="seconds to long-poll /result")
    parser.add_argument("--once", action="store_true", help="single prepare/push/result cycle (default)")
    parser.add_argument("--loop", action="store_true", help="repeat every --interval seconds")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    if args.once and args.loop:
        parser.error("--once and --loop are mutually exclusive")
    # Only remote runs need one: the server exempts loopback callers, so a client on
    # the server's own host is already authorized. Checked here rather than left to
    # the server so a genuinely remote run without a token fails now, with a reason,
    # instead of as a 401 after the raws have been read.
    if not args.token and not host_is_loopback(args.host):
        parser.error(
            f"--token is required for a remote --host ({args.host}); "
            "or set DONUT_SERVER_TOKEN"
        )
    if args.collections and not args.butler:
        parser.error("--collections only applies with --butler")

    # Resolved once, up front: --visit pins the exposure, so there is nothing to
    # re-resolve per cycle, and a bad visit fails before --loop starts rather
    # than on the first cycle. The pixels are still re-read every cycle inside
    # run_once, which is the part that mimics a real producer.
    if args.butler:
        if not args.collections:
            parser.error("--collections is required with --butler")
        from lsst.daf.butler import Butler

        butler = Butler.from_config(args.butler, writeable=False)
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
        source = resolve_from_butler(butler, args.instrument, collections, args.visit)
    else:
        source = resolve_from_files(args.raw_dir or raw_dir(), args.visit)

    if args.loop:
        print(f"looping every {args.interval}s, Ctrl-C to stop")
        while True:
            t0 = time.monotonic()
            try:
                run_once(args.host, args.token, args.calib_selector, source, args.wait)
            except requests.HTTPError as exc:
                # Transient by design: a coordinator restart answers 503 with a
                # `reason` and Retry-After, and a job whose state has moved on
                # answers 409. Dying on either would make it impossible to watch
                # the service recover across exposures, which is the whole point
                # of loop mode.
                report_transient_or_raise(exc)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, args.interval - elapsed))
    else:
        run_once(args.host, args.token, args.calib_selector, source, args.wait)


if __name__ == "__main__":
    main()
