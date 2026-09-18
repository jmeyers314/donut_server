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
