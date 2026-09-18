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

import struct
from collections.abc import Mapping
from dataclasses import dataclass

MAGIC = b"DWFS"
VERSION = 2

IMG_SUFFIX = ":img"
META_SUFFIX = ":meta"

# magic(4s), version(H), part_count(H), payload_length(Q)
_HEADER_FMT = ">4sHHQ"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# name_len(B), offset(Q), byte_count(Q) -- part name bytes follow, variable length
_DESC_FIXED_FMT = ">BQQ"
_DESC_FIXED_SIZE = struct.calcsize(_DESC_FIXED_FMT)


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
