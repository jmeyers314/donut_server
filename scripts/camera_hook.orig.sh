#!/bin/bash

set -euo pipefail

STATE_DIR="/home/ccs-ipa/fits-received"
mkdir -p "$STATE_DIR"

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
fi
