"""A stdlib-only stand-in for coordinator.plot_flight_main, for queue tests.

Same rules and the same reason as fake_coordinator: `main` must stay a
module-level function, because spawn pickles the target by reference and the
child imports *this* module rather than `coordinator` -- so no test here loads
afw, matplotlib or ts_wep, and no figure is ever drawn.

Simpler than fake_coordinator in one way that matters: it opens no shared block,
because a real plot flight is passed no block name. That asymmetry is itself
under test -- see test_a_plot_flight_needs_no_shared_block.

`script` supplies one misbehaviour per command, in order. Unlike
fake_coordinator's it applies to *every* child, since these tests exercise a
flight that keeps serving after a bad job rather than one that restarts.
"""
from __future__ import annotations

import os
import time


def main(conn, script=()) -> None:
    # Mirrors the real plot flight: own process group, so the front-end's
    # killpg reaches it.
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    try:
        conn.send({"ok": True, "event": "hello", "pid": os.getpid()})
    except (BrokenPipeError, OSError):
        # The parent is already gone. Reachable in the restart tests: a replacement
        # child is spawned by _bring_up and can reach its hello after the test has
        # torn the flight down, and the resulting traceback on stderr is pure
        # noise that makes a *real* failure harder to spot in the run output.
        return

    behaviours = list(script)
    drawn = []
    while True:
        try:
            command = conn.recv()
        except EOFError:
            break

        behaviour = behaviours.pop(0) if behaviours else None
        cmd = command.get("cmd")

        if behaviour == "crash":
            os._exit(9)
        elif behaviour == "hang":
            # Models a wedged matplotlib: alive, never touches the pipe again.
            time.sleep(300)

        # Every reply goes through this, for the same reason the hello does: a
        # teardown can close the parent end between recv() and send().
        def reply(payload) -> bool:
            try:
                conn.send(payload)
                return True
            except (BrokenPipeError, OSError):
                return False

        if cmd == "shutdown":
            reply({"ok": True})
            break
        elif cmd == "plot":
            job_id = command.get("job_id")
            if behaviour == "fail":
                # The shape plot_flight_main reports a draw failure in: an error
                # string, not an exception, so the flight keeps serving.
                if not reply({"ok": False, "error": f"fake failure drawing {job_id}"}):
                    break
                continue
            drawn.append(job_id)
            if not reply({
                "ok": True,
                "result": {
                    "job_id": job_id,
                    # The real one reports which figures it wrote; the queue only
                    # ever checks `ok`, so the contents are a placeholder.
                    "written": {"donut": {"full": 1, "thumb": 1}},
                    "skipped": [],
                    "drawn_so_far": list(drawn),
                },
            }):
                break
        elif not reply({"ok": False, "error": f"unknown cmd {cmd!r}"}):
            break
