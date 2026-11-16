"""Where the service listens, and the default every client dials.

One definition, because the port otherwise appears in four argparse defaults
across three processes plus the docs, and a drift between any two of them is a
connection refused with no hint as to which side is wrong.

Stdlib-only, and imports nothing from this package, so importing it costs
nothing and cannot pull `requests` or `lsst.afw` in behind it -- which is why a
default host lives here rather than in `client`, where it would otherwise
naturally go.

`scripts/camera_hook.sh` is the one place the port is duplicated rather than
imported: it is bash on the camera node and reads no environment by design, so
it carries the number as a constant and a comment saying to keep the two in
step.

28861 rather than 8000: the 8000s collide with everything (8000 is Django's
default and `http-alt` in practice), and a port *above* 32768 is worse than it
looks, because that is where the standard Linux ephemeral range starts -- a
listener there can find its own port held by some unrelated outbound connection
at the moment it tries to bind, which fails only on restart and looks like
nothing at all. 28861 is unassigned by IANA, below that range, and gamma/2.
"""
from __future__ import annotations

DEFAULT_PORT = 28861

# The loopback default, spelled as a URL because that is what every client's
# --host takes. Loopback and not 0.0.0.0: `check_auth` exempts a loopback peer
# from the token, so the default configuration is the one that needs no secret.
DEFAULT_HOST = f"http://127.0.0.1:{DEFAULT_PORT}"
