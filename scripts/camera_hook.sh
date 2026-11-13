#!/bin/bash
#
# The camera's per-file hook, as deployed on the ccs-ipa node.
#
# This is a *copy* for version control and review: the running script lives on
# the camera node and is invoked once per FITS file that arrives from the
# DAQ, with $FITSFILE naming the file. Edit here, copy there.
#
# Everything above the "donut server" block is the pre-existing script, which
# counts the 8 corner-sensor files of one exposure and detects completion. The
# block at the bottom is the trigger: it replaces bin/donutTrigger.py entirely,
# since the lock here already guarantees what that script's polling was for.
#
# Nothing in here reads the environment. The camera's environment is not ours
# to assume, so the two things that could come from it -- where the server is,
# and which curl -- are constants below. A stale DONUT_SERVER_* exported into
# the hook cannot misdirect a submission that way.

set -euo pipefail

STATE_DIR="/home/ccs-ipa/fits-received"
mkdir -p "$STATE_DIR"

# Loopback, so no token: the server exempts loopback callers from
# DONUT_SERVER_TOKEN, which is why this script holds no secret. If the server
# ever moves off this host, this needs an Authorization header *and* the
# exposure directory has to be visible to the server's own filesystem.
# 28861 is lsst.ts.donut_server.endpoint.DEFAULT_PORT; keep the two in step.
DONUT_SERVER_URL="http://127.0.0.1:28861/submit"

# Absolute, because PATH is part of what we do not assume. Verify with
# `command -v curl` on the node.
CURL="/usr/bin/curl"

# Generous upper bound on one exposure: this request covers the server's
# resolve, prepare and the whole fit. It is a backstop against a wedged server,
# not a cadence control.
SUBMIT_TIMEOUT=900

fitsfile="${FITSFILE:?FITSFILE is not set}"
filename=$(basename -- "$fitsfile")

# Extract exposure name from FITS filename.
if [[ ! "$filename" =~ ^(.*)_R[0-9]+_SW[0-9]+\.fits$ ]]; then
    echo "Invalid FITS filename: $filename" >&2
    exit 1
fi

exposure="${BASH_REMATCH[1]}"
state_file="$STATE_DIR/$exposure.list"

# Lock updates across concurrent processes.
exec 9>"$STATE_DIR/.lock"
flock -x 9

# Initialize state on receipt of the first file.
if [[ ! -e "$state_file" ]]; then
    jsonfile="${fitsfile%.fits}.json"
    emuimage=""

    if [[ -f "$jsonfile" ]]; then
        emuimage=$(sed -n \
            's/.*"EMUIMAGE"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
            "$jsonfile")
    else
        echo "Warning: JSON file not found: $jsonfile" >&2
    fi

    # Empty EMUIMAGE is valid.
    printf 'EMUIMAGE=%s\n' "$emuimage" > "$state_file"
else
    # Retrieve EMUIMAGE from existing state.
    IFS= read -r first_line < "$state_file"
    emuimage="${first_line#EMUIMAGE=}"
fi

# Record each distinct FITS filename only once.
if ! grep -Fxq -- "$filename" "$state_file"; then
    echo "$filename" >> "$state_file"
fi

# Count received files, excluding EMUIMAGE header.
count=$(( $(wc -l < "$state_file") - 1 ))

echo "Received $count/8: $filename EMUIMAGE=$emuimage"

# Detect completion.
if (( count == 8 )); then
    echo "Complete: $exposure EMUIMAGE=$emuimage"
    rm -f -- "$state_file"

    # ---- donut server ----
    #
    # EMUIMAGE is deliberately *not* sent. The server reads it out of the live
    # headers itself and cross-checks all 8 sensors, so this request is
    # identical for production and for a BTS replay; replay substitution is
    # switched on server-side with DONUT_SERVER_REPLAY_DIR, which only the
    # server process ever reads.

    # Release the lock before the long request. flock lives on the open file
    # description, not on this process, so the backgrounded curl below would
    # inherit fd 9 and hold the lock for the whole fit -- stalling every hook
    # of the *next* exposure at the flock above. Closed here, and again with
    # 9<&- on the curl itself so a future edit that moves this line cannot
    # quietly reintroduce the stall. The rm above stays inside the lock: it is
    # what makes "I am the 8th file" fire exactly once.
    exec 9>&-

    expdir=$(cd -- "$(dirname -- "$fitsfile")" && pwd -P)

    # One log per exposure, beside the data, so retention follows the data's.
    # The server's own records (/admin/exposures, the dashboard, its log) cover
    # everything that reaches it; this file is for what does not -- a hook that
    # never fired, or a 4xx.
    #
    # The header line doubles as a writability probe, and the /dev/null
    # fallback is load-bearing: a failed `>>` open is evaluated before exec, so
    # an unwritable directory would mean curl never runs and the exposure is
    # silently never submitted. Logging must not be able to take out the fit.
    logfile="$expdir/${exposure}_donut.log"
    printf '%s submit %s\n' "$(date -Is)" "$expdir" >>"$logfile" 2>/dev/null \
        || logfile=/dev/null

    # setsid and &: /submit returns only when the fit is DONE, and this script
    # is on the camera's critical path. setsid rather than a bare & so the
    # child survives a hook runner that enforces timeouts by killing the
    # process group, and </dev/null plus the redirects so no inherited pipe
    # keeps the invoker waiting on EOF.
    setsid "$CURL" --silent --show-error --max-time "$SUBMIT_TIMEOUT" \
        -X POST "$DONUT_SERVER_URL" \
        -H 'Content-Type: application/json' \
        --data "$(printf '{"directory":"%s"}' "$expdir")" \
        -w '\nHTTP %{http_code} in %{time_total}s\n' \
        9<&- </dev/null >>"$logfile" 2>&1 &
fi
