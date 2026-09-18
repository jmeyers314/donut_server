"""The result table's parquet round trip and JSON summary.

Uses a small synthetic table shaped like donutBlitzCornerResults --
multidimensional Zernike columns plus units -- so these run in milliseconds
without the pipeline.
"""
import io

import numpy as np
import pyarrow.parquet
import pytest
from astropy import units as u
from astropy.table import Table
from lsst.daf.butler.formatters.parquet import arrow_to_astropy

from lsst.ts.donut_server import coordinator


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    n = 6
    return Table(
        {
            "det_name": ["R00_SW0"] * 3 + ["R44_SW1"] * 3,
            "donut_id": np.arange(n),
            "group_id": ["g1", "g1", "g2", "g3", "g3", "g4"],
            "group_fit_success": [True, True, False, True, True, True],
            "snr": rng.uniform(100, 1000, n),
            "x_det": rng.uniform(0, 4000, n) * u.pix,
            "zk_deviation_ccs": rng.normal(size=(n, 27)) * u.micron,
            "zk_intrinsic_ccs": rng.normal(size=(n, 67)) * u.micron,
        }
    )


def test_parquet_round_trip_preserves_shape_units_and_values(table):
    payload = coordinator.to_parquet(table)
    back = arrow_to_astropy(pyarrow.parquet.read_table(io.BytesIO(payload)))

    assert back.colnames == table.colnames
    assert back["zk_deviation_ccs"].shape == (6, 27)
    assert back["zk_intrinsic_ccs"].shape == (6, 67)
    assert back["zk_deviation_ccs"].unit == u.micron
    assert back["x_det"].unit == u.pix
    for col in table.colnames:
        assert np.array_equal(np.asarray(back[col]), np.asarray(table[col])), col


def test_summary_counts_rows_detectors_and_groups(table):
    summary = coordinator._summarize(table)

    assert summary["n_rows"] == 6
    assert summary["n_detectors"] == 2
    assert summary["rows_per_detector"] == {"R00_SW0": 3, "R44_SW1": 3}
    assert summary["n_groups"] == 4
    # g2's single row has group_fit_success False.
    assert summary["n_groups_succeeded"] == 3
    assert summary["columns"] == table.colnames


def test_summary_is_json_safe(table):
    import json

    json.dumps(coordinator._summarize(table))


def test_image_columns_are_the_ones_that_get_dropped():
    # Guards against a rename upstream silently re-admitting 28 MB of stamps.
    assert coordinator.IMAGE_COLUMNS == ("stamp", "wf_img", "model_img")
