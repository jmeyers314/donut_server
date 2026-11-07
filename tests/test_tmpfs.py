"""Tests for the /dev/shm capacity probe.

Previously covered only through its two callers, which meant the branch that
matters most -- an unmeasurable mount -- was asserted as a side effect of a
server startup test. The probe itself is three lines of policy and worth
testing directly.
"""
from __future__ import annotations

import pytest

from lsst.ts.donut_server import tmpfs

MB = 1024 * 1024


class Refused(RuntimeError):
    """Stands in for a caller's own exception type."""


def test_an_absent_mount_is_unmeasurable_rather_than_empty():
    """macOS has no /dev/shm at all: POSIX shared memory there is kernel-backed
    with no mount to statvfs. None has to mean "cannot tell", because a 0 would
    make the service unstartable on a supported dev platform."""
    assert tmpfs.shm_free_bytes("/nonexistent/for/this/test") is None


def test_the_real_mount_measures_something_when_it_exists():
    """Guards against a probe that returns None for every input, which would
    pass every other test here while silently disabling the check in production.
    """
    free = tmpfs.shm_free_bytes()
    if free is None:
        pytest.skip(f"no {tmpfs.SHM_MOUNT} on this platform")
    assert free > 0


def test_room_enough_is_no_shortfall(monkeypatch):
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: 600 * MB)
    assert tmpfs.shortfall(513 * MB) is None


def test_exactly_enough_is_no_shortfall(monkeypatch):
    """The comparison is >=, not >: a block that fits exactly fits."""
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: 513 * MB)
    assert tmpfs.shortfall(513 * MB) is None


def test_a_measured_shortfall_reports_the_free_bytes(monkeypatch):
    """Returns the measurement, not a bool, so each caller can name both numbers
    in its own words."""
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: 64 * MB)
    assert tmpfs.shortfall(513 * MB) == 64 * MB


def test_an_unmeasurable_mount_is_no_shortfall(monkeypatch):
    """The branch this module exists to get right: unmeasurable means proceed,
    not refuse. Indistinguishable from "room enough" by design -- neither is
    grounds to stop."""
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: None)
    assert tmpfs.shortfall(513 * MB) is None


def test_require_free_raises_the_callers_own_error(monkeypatch):
    """The exception type is a parameter because a shortfall means different
    things to different callers: the producers fall back to a serial encode,
    the front-end refuses to start."""
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: 64 * MB)

    with pytest.raises(Refused) as excinfo:
        tmpfs.require_free(513 * MB, "this exposure", Refused)

    message = str(excinfo.value)
    assert "64 MB" in message and "513 MB" in message
    assert "this exposure" in message
    assert tmpfs.SHM_MOUNT in message


def test_require_free_is_silent_when_there_is_room(monkeypatch):
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: 600 * MB)
    tmpfs.require_free(513 * MB, "this exposure", Refused)


def test_require_free_is_silent_when_the_mount_is_unmeasurable(monkeypatch):
    monkeypatch.setattr(tmpfs, "shm_free_bytes", lambda *a, **k: None)
    tmpfs.require_free(513 * MB, "this exposure", Refused)
