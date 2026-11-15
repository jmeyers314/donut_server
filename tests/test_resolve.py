"""What one exposure directory resolves to, against real DAQ files.

Needs only `$DONUT_SERVER_REPLAY_DIR`, which is why these are not in
`test_daq.py`: that module compares assembled pixels against a reference, so
it skips entirely without `$DONUT_SERVER_RAW_DIR` too.

The case worth covering here is the calibration frame. A dark is taken with
tracking disabled, so it has no pointing to report, and it is the one frame
type that resolves to an incomplete answer on purpose -- `boresight` is None
and everything else is populated. `/submit` declines it on the image type;
the tests for that stub the resolve out, so nothing else exercises a real one.
"""
from __future__ import annotations

import glob
import os

import pytest

from lsst.ts.donut_server import daq, imgtypes

REPLAY_DIR = os.environ.get("DONUT_SERVER_REPLAY_DIR", "")


def _by_skippability() -> tuple[str | None, str | None]:
    """One skippable exposure directory and one fittable one, or None each.

    Classified by reading `IMGTYPE` off a single sensor rather than by
    directory name, so this keeps working on any replay tree. Only the first
    of each kind is needed, so this stops as soon as it has both.
    """
    skippable = fittable = None
    for directory in sorted(glob.glob(os.path.join(REPLAY_DIR, "*"))):
        if not os.path.isdir(directory):
            continue
        try:
            header = daq.read_header(daq.sw_paths(directory)[0])
        except Exception:
            continue
        if str(header["IMGTYPE"]).upper() in imgtypes.SKIP_IMGTYPES:
            skippable = skippable or directory
        else:
            fittable = fittable or directory
        if skippable and fittable:
            break
    return skippable, fittable


_SKIPPABLE, _FITTABLE = _by_skippability() if REPLAY_DIR else (None, None)

_WHERE = REPLAY_DIR or "$DONUT_SERVER_REPLAY_DIR (unset)"

needs_skippable = pytest.mark.skipif(
    _SKIPPABLE is None,
    reason=f"no {'/'.join(sorted(imgtypes.SKIP_IMGTYPES))} exposure in {_WHERE}",
)

needs_fittable = pytest.mark.skipif(
    _FITTABLE is None, reason=f"no fittable exposure in {_WHERE}"
)


@needs_skippable
def test_a_calibration_frame_resolves_without_a_pointing() -> None:
    """A dark is taken with tracking disabled, so it has no boresight.

    Reported as `boresight=None` rather than refused: nothing here decides
    whether an exposure is worth fitting, `/submit` does that on the image
    type.
    """
    source, image_type = daq.resolve_exposure_dir(_SKIPPABLE)

    assert image_type.upper() in imgtypes.SKIP_IMGTYPES
    assert source.boresight is None
    # Known, not unknown, which is the whole reason this returns rather than
    # raising: only the pointing is missing.
    assert source.visit
    assert source.physical_filter
    assert source.calib_time
    assert len(source.handles) == daq.EXPECTED_SENSORS


@needs_skippable
def test_the_coordinator_reports_a_calibration_frame_in_full() -> None:
    """Everything but the pointing reaches the front end.

    `/submit` needs only `imgtype` to decline the frame, but reporting the
    rest is what makes its SKIPPED body and `--dry-run` output say *which*
    exposure was declined.
    """
    from lsst.ts.donut_server import coordinator

    found = coordinator.resolve_exposure(_SKIPPABLE)

    assert found["imgtype"].upper() in imgtypes.SKIP_IMGTYPES
    assert found["boresight_ra"] is None and found["boresight_dec"] is None
    assert found["visit"] and found["physical_filter"] and found["calib_time"]
    assert len(found["detectors"]) == daq.EXPECTED_SENSORS


@needs_fittable
def test_a_fittable_frame_resolves_with_a_pointing() -> None:
    """The ordinary path: a science frame is taken with tracking on, so it has
    a boresight and nothing downstream has to cope with its absence."""
    source, image_type = daq.resolve_exposure_dir(_FITTABLE)

    assert image_type.upper() not in imgtypes.SKIP_IMGTYPES
    assert len(source.handles) == daq.EXPECTED_SENSORS
    assert source.boresight is not None
    assert source.calib_time


@needs_fittable
def test_every_sensor_agrees_about_the_exposure() -> None:
    """The five agreement checks have nothing to fire on for a real exposure.

    Asserted directly rather than trusted: the checks are only safe to make
    exact-equality tests because the 8 files really do carry identical values,
    and that is a claim about the data, not about the code.
    """
    source, _ = daq.resolve_exposure_dir(_FITTABLE)

    infos = [
        daq.obs_info(handle.header, filename=handle.path)
        for handle in source.handles.values()
    ]
    assert len({str(i.physical_filter) for i in infos}) == 1
    assert len({int(i.exposure_id) for i in infos}) == 1
    assert len({str(i.observation_type) for i in infos}) == 1
    assert len({(i.tracking_radec.ra.deg, i.tracking_radec.dec.deg) for i in infos}) == 1
