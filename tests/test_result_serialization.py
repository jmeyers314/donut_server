"""The result table's parquet round trip and JSON summary.

Uses a small synthetic table shaped like donutBlitzCornerResults --
multidimensional Zernike columns plus units -- so these run in milliseconds
without the pipeline.
"""
import io
import os

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


def test_image_columns_are_the_ones_held_out_of_the_reply():
    # Guards against a rename upstream silently re-admitting 28 MB of stamps into
    # the reply the client is blocked on. They are still written, just later.
    assert coordinator.IMAGE_COLUMNS == ("stamp", "wf_img", "model_img")


def test_stamp_table_is_written_atomically_and_keeps_the_images(table, tmp_path, monkeypatch):
    """The deferred write must leave a complete file with the images intact.

    Atomicity is the load-bearing part: the front-end serves this path with no
    completion signal from the coordinator, so a visible file has to be a whole
    one. Asserting no .tmp survives is asserting that contract.
    """
    monkeypatch.setenv("DONUT_SERVER_STAMP_DIR", str(tmp_path))
    table["stamp"] = np.zeros((len(table), 4, 4))

    path = coordinator.write_stamp_table("job-1", table)

    assert os.path.basename(path) == "job-1.parquet"
    assert os.listdir(tmp_path) == ["job-1.parquet"]

    back = arrow_to_astropy(pyarrow.parquet.read_table(path))
    assert "stamp" in back.colnames
    assert back["stamp"].shape == (len(table), 4, 4)


def test_stamp_dir_unset_is_a_loud_error(monkeypatch):
    # The write itself happens after the push has been answered, where a raise
    # reaches nobody -- so prepare calls this while a client can still be told.
    monkeypatch.delenv("DONUT_SERVER_STAMP_DIR", raising=False)
    with pytest.raises(RuntimeError, match="DONUT_SERVER_STAMP_DIR"):
        coordinator.stamp_dir()


def test_deferred_drain_clears_state_even_when_the_write_fails(table, monkeypatch):
    """A failed write must not stall the loop or strand the table.

    _run_deferred swallows its errors by design: the client already holds a 200
    for this job, so raising would kill the coordinator over stamps nobody is
    waiting on. What must not happen is the 28 MB surviving into the next job.
    """
    monkeypatch.setenv("DONUT_SERVER_STAMP_DIR", "/nonexistent-root/nope")
    coordinator._DEFERRED["job_id"] = "job-2"
    coordinator._DEFERRED["table"] = table

    coordinator._run_deferred()

    assert coordinator._DEFERRED == {}


def test_deferred_drain_with_nothing_pending_is_a_noop():
    # The loop calls this after every push, including the ones that raised before
    # a table existed.
    coordinator._DEFERRED.clear()
    coordinator._run_deferred()
