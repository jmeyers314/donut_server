# This file is part of donut_server.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Wire format for the raw-image push blob.

A single binary POST body (application/octet-stream) (JSON is too slow):

    [ fixed header ]
    [ descriptor 0 ]
    [ descriptor 1 ]
    ...
    [ part bytes concatenated ]

Descriptor offsets are relative to the start of the part-data section (i.e.
the first byte after the last descriptor), not to the start of the blob.

The framing is a generic `{part_name: bytes}` map. One blob is one whole
exposure: 8 corner sensors, each needing an image plane and a metadata pickle
(see `exposure_codec`), so 16 parts and ~294 MB.

Both of a sensor's parts are top-level, named `<sensor>:img` and
`<sensor>:meta`, so each image's descriptor points straight at its pixels: a
reader hands the codec a `memoryview` into the blob and copies nothing.
`split_parts` regroups the flat map back into `{sensor: (img, meta)}`.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

MAGIC = b"Bare Metal Blitz"
VERSION = 1

# Struct format
# > = big endian (network format)
# s# = char[#]
# B = unsigned char (1 byte)
# H = unsigned short (2 bytes)
# Q = unsigned long long (8 bytes)
# magic(16s), version(H), part_count(H), payload_length(Q)
_HEADER_FMT = ">16sHHQ"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# name_len(B), offset(Q), byte_count(Q), then that many name bytes
_DESC_FIXED_FMT = ">BQQ"
_DESC_FIXED_SIZE = struct.calcsize(_DESC_FIXED_FMT)

IMG_SUFFIX = ":img"
META_SUFFIX = ":meta"

# What the readers accept: anything they can len(), slice, and hand to
# bytes(). The front end passes a memoryview onto shared memory, which is the
# case that matters -- `parse_layout` must not copy 294 MB to read 28 bytes.
Blob = bytes | bytearray | memoryview


class ProtocolError(ValueError):
    """Raised when a blob's framing is malformed or unrecognized.

    Framing only. A part whose *contents* are wrong is `exposure_codec`'s
    business and raises `ValueError` from there instead.
    """


@dataclass(frozen=True)
class HeaderInfo:
    """The fixed header's three fields, decoded.

    Attributes
    ----------
    version : int
        Wire format version.
    part_count : int
        How many descriptors follow the header.
    payload_length : int
        Bytes in the part-data section. Counts neither the header nor the
        descriptors.
    """

    version: int
    part_count: int
    payload_length: int


@dataclass(frozen=True)
class PartLayout:
    """Where one part sits, ready to slice out of the blob.

    Attributes
    ----------
    name : str
        The part's wire name, e.g. ``R00_SW0:img``.
    offset : int
        Absolute offset from the start of the blob. The descriptor on the
        wire holds an offset relative to the part-data section; this is
        that plus the section's own start, which `parse_layout` adds, so
        no further arithmetic is needed to slice with it.
    byte_count : int
        Length of the part in bytes.

    Notes
    -----
    Small and picklable on purpose: when the blob lives in shared memory,
    a list of these is the *only* thing that needs to cross the process
    boundary.
    """

    name: str
    offset: int
    byte_count: int


def parse_header(prefix: Blob) -> HeaderInfo:
    """Decode just the fixed header.

    Parameters
    ----------
    prefix : Blob
        At least `HEADER_SIZE` bytes from the start of the blob. Longer is
        fine; only the first `HEADER_SIZE` are read.

    Returns
    -------
    HeaderInfo
        The decoded version, part count and payload length.

    Raises
    ------
    ProtocolError
        If `prefix` is shorter than `HEADER_SIZE`, the magic does not
        match, or the version is not `VERSION`.

    Notes
    -----
    Intended for early size-cap rejection: read `HEADER_SIZE` bytes off the
    incoming stream, call this, and reject before reading the rest of the
    body. That is why the header is fixed-size and comes first.
    """
    if len(prefix) < HEADER_SIZE:
        raise ProtocolError(f"prefix too short for header: got {len(prefix)}, need {HEADER_SIZE}")
    magic, version, part_count, payload_length = struct.unpack(_HEADER_FMT, prefix[:HEADER_SIZE])
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version: {version}")
    return HeaderInfo(version=version, part_count=part_count, payload_length=payload_length)


def part_name(sensor: str, suffix: str) -> str:
    """Build the wire name for one part of one sensor.

    Parameters
    ----------
    sensor : str
        Detector name, e.g. ``R00_SW0``.
    suffix : str
        `IMG_SUFFIX` or `META_SUFFIX`.

    Returns
    -------
    str
        The two concatenated, e.g. ``R00_SW0:img``.
    """
    return f"{sensor}{suffix}"


def split_parts(parts: Mapping[str, bytes]) -> dict[str, tuple[bytes, bytes]]:
    """Regroup a flat part map by sensor.

    Parameters
    ----------
    parts : Mapping[str, bytes]
        Part name to part bytes, as `unpack_blob` returns. Every name must
        end in `IMG_SUFFIX` or `META_SUFFIX`.

    Returns
    -------
    dict[str, tuple[bytes, bytes]]
        Sensor name to ``(img_blob, meta_blob)``.

    Raises
    ------
    ProtocolError
        If a name carries neither suffix, or a sensor is missing one of its
        two parts.
    """
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


def descriptor_region_size(names: Iterable[str]) -> int:
    """Bytes the descriptors for `names` will occupy.

    Parameters
    ----------
    names : Iterable[str]
        The part names that will be described.

    Returns
    -------
    int
        Size of the descriptor region alone, header excluded.

    Raises
    ------
    ProtocolError
        If any name exceeds 255 bytes once UTF-8 encoded.

    Notes
    -----
    Computable from the names alone, which is what lets a producer that
    writes parts *in place* know where the part-data section starts before
    it has any part bytes to write. See `build_framing`.
    """
    total = 0
    for name in names:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ProtocolError(f"part name too long: {name!r}")
        total += _DESC_FIXED_SIZE + len(name_bytes)
    return total


def build_framing(
    sizes: Mapping[str, int], offsets: Mapping[str, int] | None = None
) -> tuple[bytes, dict[str, int], int]:
    """Build the header and descriptors for parts of known size.

    The in-place counterpart to `pack_blob`, for a producer that writes
    each part straight into shared memory at its final wire position
    instead of building the parts and concatenating them. `pack_blob`
    materializes the whole payload twice; this copies it not at all.

    Parameters
    ----------
    sizes : Mapping[str, int]
        Part name to the number of bytes that part will occupy.
    offsets : Mapping[str, int], optional
        Each part's offset *relative to the part-data section*, for a
        caller that wants to pin its parts to slots it has sized in
        advance, before it holds any of their bytes. These need not be
        contiguous: gaps are legal, because `parse_layout` bounds-checks
        each descriptor against `payload_length` and never assumes the
        parts tile the payload. Omitted means pack them in iteration order
        with no gaps, matching `pack_blob`'s layout.

    Returns
    -------
    framing : bytes
        The header and descriptors, to be written at offset 0.
    absolute_offsets : dict[str, int]
        Where each part must then be written, absolute from the start of
        the blob.
    total_size : int
        Length of the finished blob, framing included.

    Raises
    ------
    ProtocolError
        If `offsets` is given but omits a part, or a name exceeds 255 bytes
        once UTF-8 encoded.
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
        descriptors += struct.pack(_DESC_FIXED_FMT, len(name_bytes), offsets[name], sizes[name])
        descriptors += name_bytes

    header = struct.pack(_HEADER_FMT, MAGIC, VERSION, len(names), payload_length)
    data_start = HEADER_SIZE + len(descriptors)
    absolute = {name: data_start + offsets[name] for name in names}
    return bytes(header) + bytes(descriptors), absolute, data_start + payload_length


def pack_blob(parts: Mapping[str, bytes]) -> bytes:
    """Pack parts into one wire-format blob.

    Parameters
    ----------
    parts : Mapping[str, bytes]
        Part name to part bytes. Laid out in iteration order, with no gaps.

    Returns
    -------
    bytes
        The complete blob: header, descriptors, then the part bytes
        concatenated.

    Raises
    ------
    ProtocolError
        If a name exceeds 255 bytes once UTF-8 encoded.

    Notes
    -----
    Materializes the payload twice, once in the join and once in the final
    concatenation. `build_framing` is the zero-copy alternative, for a
    producer that can write its parts in place.
    """
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


def parse_layout(blob: Blob) -> list[PartLayout]:
    """Parse the framing only, copying no payload bytes.

    Parameters
    ----------
    blob : Blob
        Sliced to exactly the received length. That length is what the
        truncation checks below are measured against, so a view onto an
        oversized block has to be sliced before it is passed.

    Returns
    -------
    list[PartLayout]
        One entry per part, carrying absolute offsets.

    Raises
    ------
    ProtocolError
        If the header is bad (see `parse_header`), a descriptor is
        truncated, the blob is shorter than `payload_length` implies, or a
        descriptor's range runs past the end of the payload.
    """
    info = parse_header(blob[:HEADER_SIZE])

    pos = HEADER_SIZE
    entries: list[tuple[str, int, int]] = []
    for _ in range(info.part_count):
        if pos + _DESC_FIXED_SIZE > len(blob):
            raise ProtocolError("truncated descriptor")
        name_len, offset, byte_count = struct.unpack(_DESC_FIXED_FMT, blob[pos : pos + _DESC_FIXED_SIZE])
        pos += _DESC_FIXED_SIZE
        name = bytes(blob[pos : pos + name_len]).decode("utf-8")
        pos += name_len
        entries.append((name, offset, byte_count))

    data_start = pos
    payload_end = data_start + info.payload_length
    if payload_end > len(blob):
        raise ProtocolError(f"blob too short: expected {payload_end} bytes, got {len(blob)}")

    layout = []
    for name, offset, byte_count in entries:
        start = data_start + offset
        if start + byte_count > payload_end:
            raise ProtocolError(f"descriptor for {name!r} exceeds payload_length")
        layout.append(PartLayout(name=name, offset=start, byte_count=byte_count))
    return layout


def unpack_blob(blob: Blob) -> dict[str, bytes]:
    """Materialize every part as its own bytes object.

    The inverse of `pack_blob`, and the copying counterpart to
    `parse_layout`.

    Parameters
    ----------
    blob : Blob
        A complete blob, as `pack_blob` returns.

    Returns
    -------
    dict[str, bytes]
        Part name to a copy of that part's bytes.

    Raises
    ------
    ProtocolError
        Whatever `parse_layout` raises for the same blob.

    Notes
    -----
    Not used by the service, which reads parts in place out of shared
    memory via `parse_layout`. Kept for an off-host consumer that has no
    shared block, and as the round-trip inverse in the tests.
    """
    view = memoryview(blob)
    return {
        part.name: bytes(view[part.offset : part.offset + part.byte_count]) for part in parse_layout(view)
    }
