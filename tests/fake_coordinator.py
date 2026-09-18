"""A stdlib-only stand-in for coordinator.coordinator_main, for lifecycle tests.

`main` must stay a module-level function: spawn pickles the target by reference,
so the child imports *this* module rather than `coordinator`, and the LSST stack
is never touched. Under the documented `python -m pytest`, sys.path carries CWD
into the child (the same reason `import protocol` works at all), and __main__ is
pytest.__main__, whose spec name ends in `.__main__`, so _fixup_main_from_name
returns early and the child does not re-run pytest.

Speaks the same protocol as the real coordinator: one hello, then exactly one
reply per command. `script` supplies one misbehaviour per command, in order, and
applies to the *first* child only -- spawn re-pickles the same script into every
replacement, so without that the replacement would reproduce the crash and no
test could observe a successful restart. Which child this is comes from a counter
in byte 0 of the shared block, the one piece of state that survives a respawn.
"""
from __future__ import annotations

import os
import struct
import time
from multiprocessing import shared_memory


def _fork_holder() -> None:
    """Fork a grandchild that survives holding the inherited fds, then die.

    This is the finding-2 regression: popen_fork closes only the fds it creates,
    so a fork worker inherits a copy of the spawn sentinel's write end and keeps
    it unreadable after the coordinator itself is gone. Detection that relies on
    the sentinel alone hangs here.
    """
    if os.fork() == 0:
        time.sleep(30)
        os._exit(0)
    os._exit(11)


def main(conn, shm_name: str, script=()) -> None:
    # Mirrors the real coordinator: own process group so the parent's killpg
    # reaches fork workers, and the shared block opened with track=False because
    # the front-end owns it.
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    shm = shared_memory.SharedMemory(name=shm_name, track=False)
    child_index = shm.buf[0]
    shm.buf[0] = child_index + 1
    conn.send({"ok": True, "event": "hello", "pid": os.getpid()})

    # Only the first child misbehaves; replacements are healthy, so a test can
    # tell a successful restart from a crash loop.
    behaviours = list(script) if child_index == 0 else []
    seen = 0
    try:
        while True:
            try:
                command = conn.recv()
            except EOFError:
                break

            seen += 1
            behaviour = behaviours.pop(0) if behaviours else None

            if behaviour == "crash":
                os._exit(9)
            elif behaviour == "hang":
                # Models a wedged fork pool: alive, never touches the pipe again.
                while True:
                    time.sleep(3600)
            elif behaviour == "fork_holder":
                _fork_holder()
            elif behaviour == "half_reply":
                # Announce a 1 KB message, send 16 bytes of it, then die. The
                # parent's recv() reads the length prefix and then hits EOF, which
                # is the truncated-message branch of _await_reply.
                fd = conn.fileno()
                os.write(fd, struct.pack("!i", 1024))
                os.write(fd, b"\x00" * 16)
                os._exit(7)
            elif isinstance(behaviour, str) and behaviour.startswith("slow:"):
                time.sleep(float(behaviour.split(":", 1)[1]))

            cmd = command.get("cmd")
            if cmd == "shutdown":
                conn.send({"ok": True})
                break
            elif cmd == "fail":
                conn.send({"ok": False, "error": "commanded failure"})
            else:
                # `seen` counts commands this *child* has handled, which is how a
                # test observes whether a prepare was replayed after a restart.
                conn.send(
                    {"ok": True, "cmd": cmd, "echo": command.get("echo"), "seen": seen}
                )
    finally:
        shm.close()


def exit_before_hello(conn, shm_name: str, script=()) -> None:
    """Models a bad env or an import failure: dies before it can say hello."""
    os._exit(3)


def hang_before_hello(conn, shm_name: str, script=()) -> None:
    """Alive but never ready -- the readiness/liveness distinction. Models a child
    partway through its ~15 s of LSST imports."""
    while True:
        time.sleep(3600)
