#!/usr/bin/env python
"""Submit one exposure to the donut server, waiting for its files to land.

This is the script the camera's per-file hook calls, with the exposure's
directory as its argument. Pass --token explicitly, or export
DONUT_SERVER_TOKEN to match whatever donutServer.py printed. All arguments are
handled by trigger.main().
"""
import sys

from lsst.ts.donut_server import trigger

if __name__ == "__main__":
    sys.exit(trigger.main())
