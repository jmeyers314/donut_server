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

    Every caller is about to create a shared block, and none of them can rely on
    the creation failing: `SharedMemory(create=True)` is only shm_open plus
    ftruncate, which reserves nothing on tmpfs and succeeds however small the
    mount is. The shortfall surfaces when a write touches a page tmpfs cannot
    back, and the kernel's only recourse there is SIGBUS.
    """
    try:
        st = os.statvfs(mount)
    except OSError:
        return None
    # f_bavail, not f_bfree: the latter counts root-only reserve a server running
    # as an ordinary user cannot have. For tmpfs they are equal, but this is also
    # the correct read if SHM_MOUNT is ever something else.
    return st.f_bavail * st.f_frsize
