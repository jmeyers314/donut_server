"""The -c/-C config override path: application, ordering, and rejection.

These drive the real `DonutBlitzCornerConfig` and build real tasks via
`build_task`, which touches no global and needs no calibs, raws or refcat
shards, so they run in well under a second.
"""
import pytest

from lsst.ts.donut_server import coordinator


def value(field, val):
    return {"kind": "value", "field": field, "value": val}


def python(text, name="<test>"):
    return {"kind": "python", "name": name, "text": text}


def test_no_overrides_builds_a_default_task():
    task, dump = coordinator.build_task([])

    assert task.config.savePlots is False
    assert dump.startswith("import ")


def test_a_value_override_is_applied():
    task, _ = coordinator.build_task([value("maxFitScatter", "2.0")])

    assert task.config.maxFitScatter == 2.0


@pytest.mark.parametrize(
    "spec, expected",
    [
        # Last write wins within one kind...
        ([value("maxFitScatter", "1.5"), value("maxFitScatter", "2.5")], 2.5),
        # ...and across the two kinds, in both directions. This is the property that
        # makes the wire format an ordered list rather than two separate fields.
        ([python("config.maxFitScatter = 3.0"), value("maxFitScatter", "2.0")], 2.0),
        ([value("maxFitScatter", "2.0"), python("config.maxFitScatter = 3.0")], 3.0),
    ],
)
def test_overrides_apply_in_list_order(spec, expected):
    task, _ = coordinator.build_task(spec)

    assert task.config.maxFitScatter == expected


def test_a_nested_field_can_be_overridden():
    task, _ = coordinator.build_task([value("donutSelector.magMax", "16.0")])

    assert task.config.donutSelector.magMax == 16.0


def test_a_multi_line_python_override_runs_and_names_its_file_in_the_traceback():
    text = (
        "import numpy as np\n"
        "config.maxFitScatter = float(np.float64(0.25))\n"
        "config.nopeNotAField = 1\n"
    )

    with pytest.raises(coordinator.ConfigOverrideError) as excinfo:
        coordinator.build_task([python(text, name="/home/op/tweaks.py")])

    # The supplied name is compiled in, so the operator's file and the offending line
    # number are what the traceback blames -- not <string>, which would be useless
    # for a 50-line override file. It hangs off __cause__, since apply_overrides
    # re-raises `from exc`; that chain is what reaches the log.
    import traceback

    cause = excinfo.value.__cause__
    rendered = "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
    assert 'File "/home/op/tweaks.py", line 3' in rendered
    # The earlier lines ran, so this really is an exec of the whole body.
    assert "nopeNotAField" in str(excinfo.value)


def test_an_unknown_field_is_rejected():
    # applyTo mutates in place and stops at the first failure, so the guarantee that
    # matters is that a rejected list cannot leave a half-applied config visible --
    # build_task touches no global, so there is nothing to check that on here beyond
    # the raise itself; ensure_prepared's reuse guard covers "leaves the previous
    # good config live".
    with pytest.raises(coordinator.ConfigOverrideError, match="nopeNotAField"):
        coordinator.build_task([value("maxFitScatter", "9.0"), value("nopeNotAField", "1")])


def test_a_wrong_type_is_rejected():
    with pytest.raises(coordinator.ConfigOverrideError, match="maxFitScatter"):
        coordinator.build_task([value("maxFitScatter", "'not a float'")])


def test_an_extra_output_connection_is_rejected_at_prepare():
    """`-c doZernikesOutput=True` must not reach the push path.

    It adds a second output connection, for which build_quantum_context builds no
    ref, and the task then dies on a bare `KeyError: 'zernikes'` *after* ~900 MB of
    raws have crossed the wire. Rejecting it here is the whole point of
    _check_connections, so this is the regression test for that.
    """
    with pytest.raises(coordinator.ConfigOverrideError, match="zernikes"):
        coordinator.build_task([value("doZernikesOutput", "True")])


def test_a_hangtimeout_under_unittimeout_is_rejected():
    # This combination makes the watchdog os._exit(1) the coordinator on every
    # ordinary job, which the front-end then replays into an unbreakable restart
    # loop that never reaches DEGRADED.
    with pytest.raises(coordinator.ConfigOverrideError, match="hangTimeout"):
        coordinator.build_task([value("hangTimeout", "1.0")])


def test_a_disabled_watchdog_is_allowed():
    # <= 0 means "watchdog off" upstream, which is a legitimate request -- the
    # hazard is a small *positive* value, so a naive "reject <= 0" would be wrong.
    task, _ = coordinator.build_task([value("hangTimeout", "0.0")])

    assert task.config.hangTimeout == 0.0


def test_the_dump_round_trips_onto_a_fresh_config():
    from lsst.ts.wep.blitz.donutBlitzCorner import DonutBlitzCornerConfig

    _, dump = coordinator.build_task([value("maxFitScatter", "2.5")])

    fresh = DonutBlitzCornerConfig()
    fresh.loadFromString(dump)

    assert fresh.maxFitScatter == 2.5


def test_no_overrides_has_a_single_key_representation():
    # The front-end sends [] rather than omitting the field, and an old client may
    # omit it; both must reuse the same task rather than rebuilding.
    assert coordinator.override_key(None) == ()
    assert coordinator.override_key([]) == ()


def test_order_changes_the_key():
    a = coordinator.override_key([value("x", "1"), value("y", "2")])
    b = coordinator.override_key([value("y", "2"), value("x", "1")])

    assert a != b


def test_require_task_refuses_before_any_prepare(monkeypatch):
    monkeypatch.setattr(coordinator, "_TASK", None)
    with pytest.raises(RuntimeError, match="no task configured"):
        coordinator.require_task()
