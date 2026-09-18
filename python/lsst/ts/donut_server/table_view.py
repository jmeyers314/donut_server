"""Turn a `donutBlitzCornerResults` parquet blob into JSON a browser can render.

Lives in the front-end, which is allowed pyarrow but not the LSST stack: pyarrow
imports nothing under `lsst.*` (checked), costs ~34 MB RSS and ~0.2 s, and leaves
the process single-threaded. The front-end never forks, and `Coord` spawns rather
than forks, so this does not touch the fork-safety story.

Two properties of the real table make a naive `JSONResponse(table.to_pydict())`
wrong rather than merely ugly, and both are measured on a real 63-donut result:

- **Every row contains NaN.** Starlette's JSONResponse renders with
  `allow_nan=False`, so a single NaN is a 500 -- and the rows with the most NaN are
  the failed fits, which are the rows an operator actually came to look at. NaN and
  +/-inf therefore become null here, and the schema says which columns held them.
- **`donut_id` is ~6.76e18, past 2^53.** JavaScript would silently round
  6761235373898405888 to ...406000. Any int64 column holding a value that big is
  emitted as strings for the whole column, so the id you read is the id on disk.
"""
from __future__ import annotations

import io
import math
from typing import Any, Optional

import pyarrow.parquet as pq

# Rows per response. The measured table is 63 rows, but n_donuts is not bounded by
# anything in the protocol, and the widest columns are 67 doubles each.
PAGE_DEFAULT = 200
PAGE_MAX = 2000

# Beyond this an integer is not exactly representable as an IEEE-754 double, which
# is the only numeric type JSON has and the only one JavaScript will parse it into.
JS_SAFE_INT = 2 ** 53


def _finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return _finite(value)
    return value


def _clean_list(value: Any) -> Any:
    if value is None:
        return None
    return [_finite(v) if isinstance(v, float) else v for v in value]


def describe(table) -> list[dict]:
    """One descriptor per column: what it is and how wide, for the UI to lay out."""
    out = []
    for field in table.schema:
        type_str = str(field.type)
        width = None
        if field.type.num_fields or type_str.startswith(("fixed_size_list", "list")):
            kind = "array"
            try:
                width = field.type.list_size
            except (AttributeError, ValueError):
                width = None
        elif type_str.startswith(("double", "float")):
            kind = "float"
        elif type_str.startswith("int") or type_str.startswith("uint"):
            kind = "int"
        elif type_str == "bool":
            kind = "bool"
        else:
            kind = "string"
        out.append({"name": field.name, "kind": kind, "type": type_str, "width": width})
    return out


def _stringify_columns(table) -> set[str]:
    """Integer columns holding a value JavaScript cannot represent exactly.

    Decided per column rather than per value, so a column never arrives as a mix of
    numbers and strings -- which would make sorting it in the UI incoherent.
    """
    unsafe = set()
    for field in table.schema:
        if not str(field.type).startswith(("int", "uint")):
            continue
        column = table[field.name]
        for chunk in column.iterchunks():
            if any(v is not None and abs(v) >= JS_SAFE_INT
                   for v in chunk.to_pylist()):
                unsafe.add(field.name)
                break
    return unsafe


def read(parquet_bytes: bytes, offset: int = 0, limit: int = PAGE_DEFAULT,
         columns: Optional[list[str]] = None) -> dict:
    """Decode a page of rows, JSON-safe.

    `columns` restricts the *payload*, not the schema: the UI needs the full column
    list to offer a picker, but should not have to ship 67-wide Zernike vectors for
    columns nobody has opened.
    """
    limit = max(1, min(PAGE_MAX, limit))
    table = pq.read_table(io.BytesIO(parquet_bytes))

    schema = describe(table)
    n_rows = table.num_rows
    offset = max(0, min(offset, n_rows))
    page = table.slice(offset, limit)

    wanted = [c["name"] for c in schema] if not columns else [
        c["name"] for c in schema if c["name"] in set(columns)]
    as_string = _stringify_columns(page)

    by_kind = {c["name"]: c["kind"] for c in schema}
    data: dict[str, list] = {}
    nan_counts: dict[str, int] = {}
    for name in wanted:
        values = page[name].to_pylist()
        if by_kind[name] == "array":
            cleaned = [_clean_list(v) for v in values]
            nan = sum(1 for v in cleaned if v is not None and any(x is None for x in v))
        elif name in as_string:
            cleaned = [None if v is None else str(v) for v in values]
            nan = 0
        else:
            cleaned = [_clean_scalar(v) for v in values]
            nan = sum(1 for orig, new in zip(values, cleaned)
                      if new is None and orig is not None)
        data[name] = cleaned
        if nan:
            nan_counts[name] = nan

    return {
        "n_rows": n_rows,
        "offset": offset,
        "returned": page.num_rows,
        "limit": limit,
        "schema": schema,
        # Named so the UI can label these "not a number", rather than an empty cell
        # that reads as missing data.
        "nan_counts": nan_counts,
        "stringified": sorted(as_string),
        "columns": wanted,
        "data": data,
    }
