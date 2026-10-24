import random

import pytest

from lsst.ts.donut_server import protocol


def test_pack_unpack_round_trip():
    rng = random.Random(0)
    images = {
        f"SENSOR_{i}": bytes(rng.randrange(256) for _ in range(1000 + i))
        for i in range(8)
    }

    blob = protocol.pack_blob(images)
    result = protocol.unpack_blob(blob)

    assert result == images


def test_split_parts_round_trip():
    rng = random.Random(1)
    sensors = [f"R0{i}_SW0" for i in range(4)]
    expected = {
        name: (
            bytes(rng.randrange(256) for _ in range(64)),
            bytes(rng.randrange(256) for _ in range(16)),
        )
        for name in sensors
    }

    parts = {}
    for name, (img, meta) in expected.items():
        parts[protocol.part_name(name, protocol.IMG_SUFFIX)] = img
        parts[protocol.part_name(name, protocol.META_SUFFIX)] = meta

    assert protocol.split_parts(protocol.unpack_blob(protocol.pack_blob(parts))) == expected


def test_split_parts_rejects_unpaired_part():
    parts = {protocol.part_name("R00_SW0", protocol.IMG_SUFFIX): b"x"}
    with pytest.raises(protocol.ProtocolError, match="missing part"):
        protocol.split_parts(parts)


def test_split_parts_rejects_unknown_suffix():
    with pytest.raises(protocol.ProtocolError, match="no recognized suffix"):
        protocol.split_parts({"R00_SW0": b"x"})


def test_unpack_accepts_bytearray():
    parts = {"a:img": b"0123", "a:meta": b"45"}
    result = protocol.unpack_blob(bytearray(protocol.pack_blob(parts)))
    assert result == parts
    assert all(isinstance(v, bytes) for v in result.values())


def test_parse_layout_offsets_are_absolute():
    """The coordinator slices shared memory by these offsets, so they must be
    absolute from the start of the blob, not relative to the data section."""
    parts = {"a:img": b"0123456789", "a:meta": b"XY", "b:img": b"zz", "b:meta": b"Q"}
    blob = protocol.pack_blob(parts)
    layout = protocol.parse_layout(blob)

    assert {p.name for p in layout} == set(parts)
    for part in layout:
        assert blob[part.offset : part.offset + part.byte_count] == parts[part.name]


def test_parse_layout_on_memoryview_of_oversized_buffer():
    """Mirrors the real path: a big reusable block sliced to what was received."""
    parts = {"a:img": b"abcd", "a:meta": b"ef"}
    blob = protocol.pack_blob(parts)
    block = bytearray(4096)
    block[: len(blob)] = blob

    view = memoryview(block)[: len(blob)]
    rebuilt = {
        p.name: bytes(view[p.offset : p.offset + p.byte_count])
        for p in protocol.parse_layout(view)
    }
    assert rebuilt == parts


def test_parse_layout_rejects_truncated_blob():
    blob = protocol.pack_blob({"a:img": b"0" * 32, "a:meta": b"1" * 8})
    with pytest.raises(protocol.ProtocolError, match="blob too short"):
        protocol.parse_layout(blob[:-4])


def test_parse_layout_rejects_truncated_descriptor():
    blob = protocol.pack_blob({"a:img": b"0" * 32, "a:meta": b"1" * 8})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_layout(blob[: protocol.HEADER_SIZE + 2])


def test_build_framing_matches_pack_blob_when_packed_tight():
    """The in-place framing and the copying packer must agree on the layout.

    `build_framing` exists so a producer can write each part straight into
    shared memory at its final position instead of concatenating; with no
    pinned offsets it must lay out exactly as `pack_blob` does, or the two
    paths would put the same exposure on the wire differently.
    """
    parts = {"R00_SW0:img": b"P" * 1000, "R00_SW0:meta": b"M" * 37}

    framing, absolute, total = protocol.build_framing(
        {name: len(payload) for name, payload in parts.items()}
    )
    blob = bytearray(total)
    blob[: len(framing)] = framing
    for name, payload in parts.items():
        blob[absolute[name] : absolute[name] + len(payload)] = payload

    assert bytes(blob) == protocol.pack_blob(parts)


def test_build_framing_tolerates_gaps_between_parts():
    """Fixed-size slots, the layout the parallel producer actually uses.

    Image slots are sized from cameraGeom before any pixel is read, so a part
    can start at a pinned offset with a hole in front of it. `parse_layout`
    bounds-checks each descriptor against `payload_length` and never assumes
    the parts tile the payload, so this is legal -- and this test is what keeps
    it legal.
    """
    sizes = {"R00_SW0:img": 1000, "R00_SW0:meta": 37}
    # 512 unused bytes between the image and its metadata.
    offsets = {"R00_SW0:img": 0, "R00_SW0:meta": 1000 + 512}

    framing, absolute, total = protocol.build_framing(sizes, offsets)
    blob = bytearray(total)
    blob[: len(framing)] = framing
    blob[absolute["R00_SW0:img"] : absolute["R00_SW0:img"] + 1000] = b"P" * 1000
    blob[absolute["R00_SW0:meta"] : absolute["R00_SW0:meta"] + 37] = b"M" * 37

    view = memoryview(blob)
    rebuilt = {
        p.name: bytes(view[p.offset : p.offset + p.byte_count])
        for p in protocol.parse_layout(view)
    }
    assert rebuilt == {"R00_SW0:img": b"P" * 1000, "R00_SW0:meta": b"M" * 37}


def test_build_framing_requires_an_offset_for_every_part():
    with pytest.raises(protocol.ProtocolError, match="no offset given"):
        protocol.build_framing({"a:img": 8, "a:meta": 4}, {"a:img": 0})
