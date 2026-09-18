#!/usr/bin/env python
"""Run the donut_server test client.

Pass --token explicitly, or export DONUT_SERVER_TOKEN to match whatever
donutServer.py printed. All arguments are handled by client.main().
"""
from lsst.ts.donut_server import client

if __name__ == "__main__":
    client.main()
