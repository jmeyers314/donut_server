"""How much room `/dev/shm` has, for whoever is about to write a block there.

Imported by both processes and by the producers: the front-end before it
creates a flight's shared block, `daq` before it creates one of its own. Stdlib
only, and deliberately so -- `server.py` must not import any part of the LSST
stack, and `daq` must not pull numpy in ahead of the env block that clamps
OpenBLAS to one thread.

Its own module rather than a corner of the wire format, which is what it used
to be: the two have no relationship beyond both being things a stack-free
producer needs.
"""
from __future__ import annotations

import os

# Where a tmpfs-backed shm_open puts its blocks, when there is one. Checked for
# existence rather than assumed: this is the Linux layout, and on macOS --
# a supported dev platform -- POSIX shared memory is kernel-backed with no
# mount to interrogate at all, so the capacity check is skipped there.
# The path is not configurable: CPython prefixes the block name with "/" and
# hands it to shm_open, which resolves it under this mount wherever it exists.
SHM_MOUNT = "/dev/shm"


def shm_free_bytes(mount: str = SHM_MOUNT) -> int | None:
    """Bytes available on `mount`, or None if there is nothing to measure.

    None covers both "no such mount" (macOS, where POSIX shared memory is
    kernel-backed) and a statvfs that fails for any reason: in neither case do we
    know the capacity, and a guess would either refuse a working server or
    promise capacity that is not there.
    """
    try:
        st = os.statvfs(mount)
    except OSError:
        return None
    # f_bavail, not f_bfree: the latter counts root-only reserve a server running
    # as an ordinary user cannot have. For tmpfs they are equal, but this is also
    # the correct read if SHM_MOUNT is ever something else.
    return st.f_bavail * st.f_frsize


def require_free(
    needed: int, what: str, error: type[Exception], mount: str = SHM_MOUNT
) -> None:
    """Raise `error` unless `mount` can back `needed` bytes.

    Call this *before* creating the block, never after: nobody can rely on the
    creation failing. `SharedMemory(create=True)` is only shm_open plus
    ftruncate, which reserves nothing on tmpfs and succeeds however small the
    mount is, and the mmap succeeds too. The shortfall surfaces only when a
    write touches a page tmpfs cannot back, where the kernel's sole recourse is
    SIGBUS -- so the process dies mid-job leaving nothing but "Bus error".

    A floor, not a reservation: nothing stops another process taking the space
    between here and the write. It catches the standing misconfiguration -- a
    64 MB container default against a 513 MB block -- rather than the race.

    Unmeasurable capacity means proceed, not refuse; see `shm_free_bytes`.

    `error` is the caller's own exception type because the callers differ in
    what a shortfall means: the producers fall back to encoding serially, while
    the front-end has nothing to fall back to and refuses to start.
    """
    free = shm_free_bytes(mount)
    if free is None or free >= needed:
        return
    mb = 1024 * 1024
    raise error(
        f"{mount} has {free / mb:.0f} MB free but {what} needs {needed / mb:.0f} MB"
    )
