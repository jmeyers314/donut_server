"""Wire format for the raw-image push blob.

A single binary POST body (application/octet-stream), not base64, not JSON:

    [ fixed header ][ descriptor 0 ][ descriptor 1 ]...[ part bytes concatenated ]

The header is fixed-size and comes first specifically so a server can read
just the header off a stream, check `payload_length` against a size cap, and
reject oversized/malformed pushes before reading the rest of the body.

Descriptor offsets are relative to the start of the part-data section (i.e.
the first byte after the last descriptor), not to the start of the blob.

The framing is a generic `{part_name: bytes}` map. An exposure needs two parts
per sensor -- the image plane and a metadata pickle (see `exposure_codec`) --
carried as two separate top-level parts named `<sensor>:img` and
`<sensor>:meta`. Two flat parts rather than one nested blob so each 37 MB image
lands as its own slice and needs no second split copy. `split_parts` reassembles
them into `{sensor: (img, meta)}`.
"""
from __future__ import annotations

import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass

MAGIC = b"DWFS"
VERSION = 2

# Image types that are never worth a wavefront fit. Calibration frames have no
# stars, so they would burn a full flight to arrive at an empty donut catalog.
SKIP_IMGTYPES = frozenset({"DARK", "BIAS", "FLAT"})

# Types a wavefront fit is meaningful for. Anything else is processed but
# flagged, the same way run_sequence.py flags it.
EXPECTED_IMGTYPES = frozenset({"CWFS", "ACQ", "SCIENCE", "OBJECT"})

# Both here, with the wire format, for the same reason `shm_free_bytes` is: the
# front end decides whether to skip a frame, and it imports no part of the LSST
# stack -- so the sets cannot live beside the header reading that produces the
# image type they are compared against.

# Where a tmpfs-backed shm_open puts its blocks, when there is one. Checked for
# existence rather than assumed: this is the Linux layout, and on macOS --
# a supported dev platform -- POSIX shared memory is kernel-backed with no
# mount to interrogate at all, so the capacity check is skipped there.
# The path is not configurable: CPython prefixes the block name with "/" and
# hands it to shm_open, which resolves it under this mount wherever it exists.
SHM_MOUNT = "/dev/shm"

IMG_SUFFIX = ":img"
META_SUFFIX = ":meta"

# magic(4s), version(H), part_count(H), payload_length(Q)
_HEADER_FMT = ">4sHHQ"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# name_len(B), offset(Q), byte_count(Q) -- part name bytes follow, variable length
_DESC_FIXED_FMT = ">BQQ"
_DESC_FIXED_SIZE = struct.calcsize(_DESC_FIXED_FMT)


def shm_free_bytes(mount: str = SHM_MOUNT) -> int | None:
    """Bytes available on `mount`, or None if there is nothing to measure.

    None covers both "no such mount" (macOS, where POSIX shared memory is
    kernel-backed) and a statvfs that fails for any reason: in neither case do we
    know the capacity, and a guess would either refuse a working server or
    promise capacity that is not there.

    Lives here rather than in `server.py` -- where it started and where
    `check_shm_capacity` still calls it -- because the producer side needs the
    same check before it writes a block, and `server.py` imports FastAPI.
    """
    try:
        st = os.statvfs(mount)
    except OSError:
        return None
    # f_bavail, not f_bfree: the latter counts root-only reserve a server running
    # as an ordinary user cannot have. For tmpfs they are equal, but this is also
    # the correct read if SHM_MOUNT is ever something else.
    return st.f_bavail * st.f_frsize


class ProtocolError(ValueError):
    """Raised on malformed/unrecognized blobs."""


@dataclass(frozen=True)
class HeaderInfo:
    version: int
    part_count: int
    payload_length: int


@dataclass(frozen=True)
class PartLayout:
    """Where one part sits, as an absolute offset from the start of the blob.

    Small and picklable on purpose: when the blob lives in shared memory, a
    list of these is the *only* thing that needs to cross the process boundary.
    """

    name: str
    offset: int
    byte_count: int


def parse_header(prefix: bytes) -> HeaderInfo:
    """Decode just the fixed header. `prefix` must be >= HEADER_SIZE bytes.

    Intended for early size-cap rejection: read HEADER_SIZE bytes off the
    incoming stream, call this, and reject before reading the rest of the body.
    """
    if len(prefix) < HEADER_SIZE:
        raise ProtocolError(
            f"prefix too short for header: got {len(prefix)}, need {HEADER_SIZE}"
        )
    magic, version, part_count, payload_length = struct.unpack(
        _HEADER_FMT, prefix[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version: {version}")
    return HeaderInfo(version=version, part_count=part_count, payload_length=payload_length)


def part_name(sensor: str, suffix: str) -> str:
    """Wire name for one part of `sensor`. `suffix` is IMG_SUFFIX or META_SUFFIX."""
    return f"{sensor}{suffix}"


def split_parts(parts: Mapping[str, bytes]) -> dict[str, tuple[bytes, bytes]]:
    """Regroup a flat part map into {sensor: (img_blob, meta_blob)}."""
    grouped: dict[str, dict[str, bytes]] = {}
    for name, payload in parts.items():
        for suffix in (IMG_SUFFIX, META_SUFFIX):
            if name.endswith(suffix):
                grouped.setdefault(name[: -len(suffix)], {})[suffix] = payload
                break
        else:
            raise ProtocolError(f"part {name!r} has no recognized suffix")

    result = {}
    for sensor, found in grouped.items():
        missing = [s for s in (IMG_SUFFIX, META_SUFFIX) if s not in found]
        if missing:
            raise ProtocolError(f"sensor {sensor!r} missing part(s): {missing}")
        result[sensor] = (found[IMG_SUFFIX], found[META_SUFFIX])
    return result


def descriptor_region_size(names) -> int:
    """Bytes the descriptors for `names` will occupy.

    Computable from the names alone, which is what lets a producer that writes
    parts *in place* know where the part-data section starts before it has any
    part bytes to write. See `build_framing`.
    """
    total = 0
    for name in names:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ProtocolError(f"part name too long: {name!r}")
        total += _DESC_FIXED_SIZE + len(name_bytes)
    return total


def build_framing(sizes: Mapping[str, int], offsets: Mapping[str, int] | None = None):
    """The header+descriptor prefix for parts of known size, and their offsets.

    Returns `(framing, absolute_offsets, total_size)`: the bytes to write at the
    start of the blob, where each part must then be written (absolute, from the
    start of the blob), and how long the whole blob is.

    This is the in-place counterpart to `pack_blob`, for a producer that writes
    each part straight into shared memory at its final wire position instead of
    building the parts and concatenating them. `pack_blob` copies the whole
    payload twice; this copies it not at all.

    `offsets` optionally pins each part's offset *relative to the part-data
    section*, for a caller that wants fixed-size slots (e.g. one per sensor,
    sized from cameraGeom before any pixel is read). Gaps between parts are
    legal -- `parse_layout` bounds-checks each descriptor against
    `payload_length` and never assumes the parts tile the payload -- so the
    pinned offsets need not be contiguous. Omitted means pack them in iteration
    order with no gaps, matching `pack_blob`'s layout.
    """
    names = list(sizes)
    if offsets is None:
        offsets = {}
        cursor = 0
        for name in names:
            offsets[name] = cursor
            cursor += sizes[name]
        payload_length = cursor
    else:
        missing = [name for name in names if name not in offsets]
        if missing:
            raise ProtocolError(f"no offset given for part(s): {missing}")
        # The payload must *cover* every part, including any gap before it.
        payload_length = max(offsets[name] + sizes[name] for name in names)

    descriptors = bytearray()
    for name in names:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ProtocolError(f"part name too long: {name!r}")
        descriptors += struct.pack(
            _DESC_FIXED_FMT, len(name_bytes), offsets[name], sizes[name]
        )
        descriptors += name_bytes

    header = struct.pack(_HEADER_FMT, MAGIC, VERSION, len(names), payload_length)
    data_start = HEADER_SIZE + len(descriptors)
    absolute = {name: data_start + offsets[name] for name in names}
    return bytes(header) + bytes(descriptors), absolute, data_start + payload_length


def pack_blob(parts: Mapping[str, bytes]) -> bytes:
    """Pack {part_name: bytes} into the wire format."""
    names = list(parts.keys())

    descriptors = bytearray()
    raw_offset = 0
    for name in names:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ProtocolError(f"part name too long: {name!r}")
        byte_count = len(parts[name])
        descriptors += struct.pack(_DESC_FIXED_FMT, len(name_bytes), raw_offset, byte_count)
        descriptors += name_bytes
        raw_offset += byte_count

    payload_length = raw_offset
    header = struct.pack(_HEADER_FMT, MAGIC, VERSION, len(names), payload_length)

    raw_data = b"".join(parts[name] for name in names)
    return bytes(header) + bytes(descriptors) + raw_data


def parse_layout(blob) -> list[PartLayout]:
    """Parse the framing only, copying no payload bytes.

    `blob` is any buffer (bytes, bytearray, or a memoryview onto shared memory)
    sliced to exactly the received length -- the length is what the truncation
    checks below are measured against.
    """
    info = parse_header(blob[:HEADER_SIZE])

    pos = HEADER_SIZE
    entries: list[tuple[str, int, int]] = []
    for _ in range(info.part_count):
        if pos + _DESC_FIXED_SIZE > len(blob):
            raise ProtocolError("truncated descriptor")
        name_len, offset, byte_count = struct.unpack(
            _DESC_FIXED_FMT, blob[pos : pos + _DESC_FIXED_SIZE]
        )
        pos += _DESC_FIXED_SIZE
        name = bytes(blob[pos : pos + name_len]).decode("utf-8")
        pos += name_len
        entries.append((name, offset, byte_count))

    data_start = pos
    payload_end = data_start + info.payload_length
    if payload_end > len(blob):
        raise ProtocolError(
            f"blob too short: expected {payload_end} bytes, got {len(blob)}"
        )

    layout = []
    for name, offset, byte_count in entries:
        start = data_start + offset
        if start + byte_count > payload_end:
            raise ProtocolError(f"descriptor for {name!r} exceeds payload_length")
        layout.append(PartLayout(name=name, offset=start, byte_count=byte_count))
    return layout


def unpack_blob(blob) -> dict[str, bytes]:
    """Inverse of pack_blob, materializing every part as its own bytes object.

    The service itself reads parts in place out of shared memory via
    `parse_layout`; this is the copying convenience form.
    """
    view = memoryview(blob)
    return {
        part.name: bytes(view[part.offset : part.offset + part.byte_count])
        for part in parse_layout(view)
    }
